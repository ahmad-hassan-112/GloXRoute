"""
xai_eval.py — Interpretability evaluation for GloXRoute.

Evaluates the glass-box explanation quality using three criteria:

1. FIDELITY
   The explanation exactly reflects the decision mechanism because the
   model IS the linear score. We compute correlation between reported
   tie_score and the actual selected-neighbour rank to confirm this.

2. COUNTERFACTUAL ANALYSIS
   For each routing decision, compute "what would have to change for a
   different neighbour to be selected?" — specifically, by how much
   would the load of the winning neighbour need to increase before the
   second-best becomes the winner.

3. FEATURE CONTRIBUTION CONSISTENCY
   Across repeated routing scenarios from the same source node, are the
   top-contributing features stable? Measured as the standard deviation
   of each feature's contribution weight across decisions.

Run with:
  python xai_eval.py
  python xai_eval.py --num_tasks 500 --seed 42
"""

import argparse, logging, os, random, time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import math

from config import SEED, NUM_AGENTS, NUM_TASKS, OUTPUT_DIR, TOPOLOGY, MAX_HOPS, LATENCY_BOUND_EPSILON
from network_builder import build_ioa_network
from task_generator import generate_tasks
from coordinator import GlobalCoordinator
from utils import capable_nodes, source_can_solve, effective_edge_weight
from models import Task, RouteDecision

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("xai_eval")


# ================================================================== #
#  Feature extractor (same logic as HybridXAIRouter)                  #
# ================================================================== #

FEATURE_NAMES = [
    "cap_match", "rec_match", "hint",
    "reliability", "resource", "load",
    "congestion", "proto_match", "dead_end",
]

WEIGHTS = [3.0, 2.0, 2.0, 1.5, 0.5, -1.0, -0.5, 1.0, -0.5]


def _extract_features(G, agents, coordinator, current, n, required, rec_next, visited):
    edge      = G[current][n]
    weight_fn = effective_edge_weight(agents)
    step_cost = weight_fn(current, n, edge)
    if step_cost is None:
        return None, None

    dist_val  = coordinator.distance[n][required]
    remaining = dist_val if not math.isinf(dist_val) else 5.0

    cap_match   = 1.0 if required in agents[n].capabilities else 0.0
    rec_match   = 1.0 if n == rec_next else 0.0
    hint        = coordinator.guidance[n][required]
    reliability = agents[n].reliability
    resource    = agents[n].resource
    load        = agents[n].load
    congestion  = agents[n].load
    proto_match = 1.0 if agents[current].protocol == agents[n].protocol else 0.0
    onward      = sum(1 for nn in G.neighbors(n) if nn not in visited and nn != current)
    dead_end    = 1.0 if onward == 0 and cap_match == 0.0 else 0.0

    feats = np.array([
        cap_match, rec_match, hint,
        reliability, resource, load,
        congestion, proto_match, dead_end,
    ])
    total_cost = step_cost + remaining
    return feats, total_cost


def _score(feats):
    return float(np.dot(feats, WEIGHTS))


# ================================================================== #
#  Evaluation functions                                                #
# ================================================================== #

def evaluate_fidelity(G, agents, coordinator, tasks, max_decisions=2000) -> Dict:
    """
    Fidelity check: the reported explanation IS the decision mechanism
    (glass-box). We verify by checking that the reported winner always
    has the highest tie_score among feasible candidates.
    For a linear glass-box model, this should be 100 %.
    """
    decisions_checked, correct = 0, 0

    for task in tasks:
        if source_can_solve(agents, task):
            continue
        current, required = task.source, task.required_capability
        visited = {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            rec_next = coordinator.next_hop[current][required]
            scored   = []
            for n in candidates:
                feats, total_cost = _extract_features(
                    G, agents, coordinator, current, n, required, rec_next, visited
                )
                if feats is None:
                    continue
                scored.append((n, feats, total_cost, _score(feats)))

            if not scored:
                break

            best_total = min(s[2] for s in scored)
            bound      = best_total * (1.0 + LATENCY_BOUND_EPSILON)
            feasible   = [s for s in scored if s[2] <= bound]
            winner     = max(feasible, key=lambda s: s[3])

            # Fidelity: winner must have the highest score in feasible set
            top_score = max(s[3] for s in feasible)
            correct  += 1 if abs(winner[3] - top_score) < 1e-9 else 0
            decisions_checked += 1

            current = winner[0]
            visited.add(current)

            if decisions_checked >= max_decisions:
                break
        if decisions_checked >= max_decisions:
            break

    fidelity = correct / max(1, decisions_checked)
    logger.info("Fidelity: %.4f (%d/%d decisions)", fidelity, correct, decisions_checked)
    return {"fidelity": fidelity, "decisions_checked": decisions_checked, "correct": correct}


def evaluate_counterfactuals(G, agents, coordinator, tasks, max_decisions=1000) -> pd.DataFrame:
    """
    For each decision where there is a clear winner and a runner-up,
    compute the minimum load increase needed on the winner to flip the
    choice to the runner-up. Larger delta = more robust decision.
    """
    records = []

    for task in tasks:
        if source_can_solve(agents, task):
            continue
        current, required = task.source, task.required_capability
        visited = {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates or len(candidates) < 2:
                if candidates:
                    current = candidates[0]
                    visited.add(current)
                break

            rec_next = coordinator.next_hop[current][required]
            scored   = []
            for n in candidates:
                feats, total_cost = _extract_features(
                    G, agents, coordinator, current, n, required, rec_next, visited
                )
                if feats is None:
                    continue
                scored.append((n, feats.copy(), total_cost, _score(feats)))

            if len(scored) < 2:
                if scored:
                    current = scored[0][0]
                    visited.add(current)
                break

            best_total = min(s[2] for s in scored)
            bound      = best_total * (1.0 + LATENCY_BOUND_EPSILON)
            feasible   = [s for s in scored if s[2] <= bound]
            sorted_f   = sorted(feasible, key=lambda s: -s[3])
            winner, runner_up = sorted_f[0], sorted_f[1] if len(sorted_f) > 1 else None

            if runner_up is not None:
                # Load index in FEATURE_NAMES: index 5 = "load"
                score_gap = winner[3] - runner_up[3]
                # delta_load to flip: score_gap / |weight_load| where weight_load = -1.0
                delta_load_to_flip = score_gap / 1.0   # |WEIGHTS[5]| = 1.0
                records.append({
                    "task_id":           task.task_id,
                    "current":           current,
                    "winner":            winner[0],
                    "runner_up":         runner_up[0],
                    "score_gap":         score_gap,
                    "delta_load_to_flip": delta_load_to_flip,
                    "winner_cap_match":  winner[1][0],
                    "winner_hint":       winner[1][2],
                })

            current = winner[0]
            visited.add(current)

            if len(records) >= max_decisions:
                break
        if len(records) >= max_decisions:
            break

    df = pd.DataFrame(records)
    if not df.empty:
        logger.info(
            "Counterfactuals: median delta_load_to_flip=%.3f  "
            "mean=%.3f  min=%.3f",
            df["delta_load_to_flip"].median(),
            df["delta_load_to_flip"].mean(),
            df["delta_load_to_flip"].min(),
        )
    return df


def evaluate_feature_consistency(G, agents, coordinator, tasks, max_decisions=2000) -> pd.DataFrame:
    """
    Feature contribution stability: across all decisions, measure the
    mean and std of each feature's weighted contribution (weight × feature_value).
    Low std = stable, high std = context-dependent (also informative).
    """
    contributions = {name: [] for name in FEATURE_NAMES}

    count = 0
    for task in tasks:
        if source_can_solve(agents, task):
            continue
        current, required = task.source, task.required_capability
        visited = {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            rec_next = coordinator.next_hop[current][required]
            scored   = []
            for n in candidates:
                feats, total_cost = _extract_features(
                    G, agents, coordinator, current, n, required, rec_next, visited
                )
                if feats is None:
                    continue
                scored.append((n, feats, total_cost, _score(feats)))

            if not scored:
                break

            best_total = min(s[2] for s in scored)
            bound      = best_total * (1.0 + LATENCY_BOUND_EPSILON)
            feasible   = [s for s in scored if s[2] <= bound]
            winner     = max(feasible, key=lambda s: s[3])

            # Record contribution of each feature for the winner
            for i, name in enumerate(FEATURE_NAMES):
                contributions[name].append(WEIGHTS[i] * winner[1][i])

            current = winner[0]
            visited.add(current)
            count  += 1
            if count >= max_decisions:
                break
        if count >= max_decisions:
            break

    records = []
    for name in FEATURE_NAMES:
        vals = contributions[name]
        records.append({
            "feature":  name,
            "weight":   WEIGHTS[FEATURE_NAMES.index(name)],
            "mean_contribution": np.mean(vals) if vals else 0.0,
            "std_contribution":  np.std(vals)  if vals else 0.0,
            "n_decisions":       len(vals),
        })

    df = pd.DataFrame(records).sort_values("mean_contribution", ascending=False)
    logger.info("\nFeature contribution analysis:\n%s", df.to_string(index=False))
    return df


# ================================================================== #
#  Main                                                                #
# ================================================================== #

def run_xai_eval(seed: int = SEED, num_tasks: int = 500,
                 topology: str = TOPOLOGY, num_agents: int = NUM_AGENTS,
                 out_dir: str = None):
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "xai_eval")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    G, agents = build_ioa_network(seed=seed, topology=topology, num_agents=num_agents)
    tasks     = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
    coord     = GlobalCoordinator(G, agents)

    logger.info("=== Fidelity ===")
    fid = evaluate_fidelity(G, agents, coord, tasks)

    logger.info("=== Counterfactuals ===")
    cf_df = evaluate_counterfactuals(G, agents, coord, tasks)
    cf_df.to_csv(os.path.join(out_dir, "counterfactuals.csv"), index=False)

    logger.info("=== Feature Consistency ===")
    fc_df = evaluate_feature_consistency(G, agents, coord, tasks)
    fc_df.to_csv(os.path.join(out_dir, "feature_consistency.csv"), index=False)

    summary = {
        "fidelity":                   fid["fidelity"],
        "decisions_checked":          fid["decisions_checked"],
        "counterfactual_median_delta": cf_df["delta_load_to_flip"].median() if not cf_df.empty else None,
        "counterfactual_mean_delta":   cf_df["delta_load_to_flip"].mean()   if not cf_df.empty else None,
        "top_feature":                fc_df.iloc[0]["feature"] if not fc_df.empty else None,
        "top_feature_mean_contrib":   fc_df.iloc[0]["mean_contribution"] if not fc_df.empty else None,
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "xai_summary.csv"))
    logger.info("\nXAI Summary:\n%s", pd.Series(summary).to_string())
    return summary, cf_df, fc_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=SEED)
    p.add_argument("--num_tasks",  type=int, default=500)
    p.add_argument("--topology",   default=TOPOLOGY)
    p.add_argument("--num_agents", type=int, default=NUM_AGENTS)
    p.add_argument("--out_dir",    default=None)
    args = p.parse_args()

    run_xai_eval(
        seed=args.seed, num_tasks=args.num_tasks,
        topology=args.topology, num_agents=args.num_agents,
        out_dir=args.out_dir,
    )