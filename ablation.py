"""
ablation.py — Ablation study for GloXRoute.

Runs 5 ablated variants of HybridXAI to show which components matter:
  full          — full HybridXAI (reference)
  no_global     — remove global coordinator guidance (hint=0, rec_match=0)
  no_capability — remove capability_match term
  no_load       — remove load/congestion terms
  no_reliability— remove reliability term
  no_protocol   — remove protocol compatibility term

Run with:
  python ablation.py
  python ablation.py --multiseed
"""

import argparse, logging, os, random, time
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
from collections import OrderedDict

from config import TIE_BREAK_TOP_K
from config import (
    SEED, SEEDS, NUM_TASKS, OUTPUT_DIR, COORDINATOR_UPDATE_INTERVAL,
    TOPOLOGY, NUM_AGENTS, LATENCY_BOUND_EPSILON, ROUTE_CACHE_MAXSIZE, MAX_HOPS,
)
from models import Agent, Task, RouteDecision
from network_builder import build_ioa_network
from task_generator import generate_tasks
from coordinator import GlobalCoordinator
from metrics import evaluate_route
from utils import capable_nodes, source_can_solve, effective_edge_weight
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ablation")


# ================================================================== #
#  Configurable HybridXAI for ablation                                #
# ================================================================== #

class AblatedHybridXAIRouter:
    """
    HybridXAI with individual components switchable for ablation.
    use_global     : include global coordinator guidance
    use_capability : include capability_match term
    use_load       : include load/congestion penalty
    use_reliability: include reliability bonus
    use_protocol   : include protocol compatibility bonus
    """

    def __init__(self, coordinator: GlobalCoordinator, name: str,
                 use_global: bool = True,
                 use_capability: bool = True,
                 use_load: bool = True,
                 use_reliability: bool = True,
                 use_protocol: bool = True):
        self.coordinator    = coordinator
        self.name           = name
        self.use_global     = use_global
        self.use_capability = use_capability
        self.use_load       = use_load
        self.use_reliability= use_reliability
        self.use_protocol   = use_protocol
        self.route_cache    = OrderedDict()

    def route(self, G, agents, task) -> RouteDecision:
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited     = [current], {current}
        explanations      = []
        weight_fn         = effective_edge_weight(agents)

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            rec_next = (self.coordinator.next_hop[current][required]
                        if self.use_global else None)

            rows = []
            for n in candidates:
                edge      = G[current][n]
                step_cost = weight_fn(current, n, edge)
                if step_cost is None:
                    continue

                dist_val  = self.coordinator.distance[n][required] if self.use_global else float("inf")
                if math.isinf(dist_val):
                    remaining = 5.0   # neutral fallback when global disabled
                else:
                    remaining = dist_val

                # Score components — zeroed out according to ablation flags
                cap_match   = (1.0 if required in agents[n].capabilities else 0.0) if self.use_capability else 0.0
                rec_match   = (1.0 if n == rec_next else 0.0) if self.use_global else 0.0
                hint        = (self.coordinator.guidance[n][required]) if self.use_global else 0.0
                reliability = agents[n].reliability if self.use_reliability else 0.5
                resource    = agents[n].resource
                load        = agents[n].load if self.use_load else 0.0
                congestion  = agents[n].load if self.use_load else 0.0
                proto_match = (1.0 if agents[current].protocol == agents[n].protocol else 0.0) if self.use_protocol else 0.0

                onward   = sum(1 for nn in G.neighbors(n) if nn not in visited and nn != current)
                dead_end = 1.0 if onward == 0 and cap_match == 0.0 else 0.0

                rows.append({
                    "node": n,
                    "total_cost": step_cost + remaining,
                    "tie_score": (
                        3.0 * cap_match
                        + 2.0 * rec_match
                        + 2.0 * hint
                        + 1.5 * reliability
                        + 0.5 * resource
                        - 1.0 * load
                        - 0.5 * congestion
                        + 1.0 * proto_match
                        - 0.5 * dead_end
                    ),
                })

            if not rows:
                break

            best_total = min(r["total_cost"] for r in rows)
            bound      = best_total * (1.0 + LATENCY_BOUND_EPSILON)
            feasible   = [r for r in rows if r["total_cost"] <= bound]
            # On low-degree graphs the percentage window can admit only 1
            # candidate; top-up with the next-cheapest options so the score
            # terms are actually exercised (see TIE_BREAK_TOP_K in config.py).
            if len(feasible) < TIE_BREAK_TOP_K:
                feasible = sorted(rows, key=lambda r: r["total_cost"])[:TIE_BREAK_TOP_K]
            w          = max(feasible, key=lambda r: r["tie_score"])
            best_n     = w["node"]

            current = best_n
            path.append(current)
            visited.add(current)
            explanations.append(f"from={current-1}->to={best_n}; tie={w['tie_score']:.3f}")

        return RouteDecision(path=path, explanation=" | ".join(explanations), extra_overhead=0.1)


# ================================================================== #
#  Ablation runner                                                     #
# ================================================================== #

ABLATION_VARIANTS = [
    dict(name="GloXRoute_Full",         use_global=True,  use_capability=True,  use_load=True,  use_reliability=True,  use_protocol=True),
    dict(name="Ablation_NoGlobal",      use_global=False, use_capability=True,  use_load=True,  use_reliability=True,  use_protocol=True),
    dict(name="Ablation_NoCapability",  use_global=True,  use_capability=False, use_load=True,  use_reliability=True,  use_protocol=True),
    dict(name="Ablation_NoLoad",        use_global=True,  use_capability=True,  use_load=False, use_reliability=True,  use_protocol=True),
    dict(name="Ablation_NoReliability", use_global=True,  use_capability=True,  use_load=True,  use_reliability=False, use_protocol=True),
    dict(name="Ablation_NoProtocol",    use_global=True,  use_capability=True,  use_load=True,  use_reliability=True,  use_protocol=False),
]


def run_ablation_seed(seed: int, topology: str, num_agents: int, num_tasks: int,
                      run_dir: str) -> pd.DataFrame:
    os.makedirs(run_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    G, agents = build_ioa_network(seed=seed, topology=topology, num_agents=num_agents)
    tasks     = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
    coord     = GlobalCoordinator(G, agents)

    routers = [AblatedHybridXAIRouter(coord, **v) for v in ABLATION_VARIANTS]

    all_rows = []
    for router in routers:
        logger.info("  Ablation variant: %s", router.name)
        counter = 0
        for task in tasks:
            counter += 1
            if counter % COORDINATOR_UPDATE_INTERVAL == 0:
                coord.update_guidance()
            t0  = time.perf_counter()
            dec = router.route(G, agents, task)
            dt  = time.perf_counter() - t0
            row = evaluate_route(G=G, agents=agents, task=task,
                                 method_name=router.name, path=dec.path,
                                 extra_overhead=dec.extra_overhead,
                                 explanation=dec.explanation or None,
                                 decision_time=dt)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(run_dir, "ablation_raw.csv"), index=False)

    # Summarise
    routing_df = df[~df["source_capable"]].copy()
    agg = routing_df.groupby("method").agg(
        success_rate          = ("success", "mean"),
        avg_latency           = ("latency", "mean"),
        avg_effective_latency = ("effective_latency", "mean"),
        avg_hops              = ("hops", "mean"),
        avg_overhead          = ("communication_overhead", "mean"),
        avg_path_stretch      = ("path_stretch", lambda x: x.dropna().mean()),
        avg_load_on_path      = ("avg_load_on_path", "mean"),
        protocol_mismatch_rate= ("protocol_mismatch_rate", "mean"),
    ).reset_index()

    agg["routing_score"] = (
        0.30 / (1 + agg["avg_latency"])
        + 0.25 / (1 + agg["avg_path_stretch"])
        + 0.20 / (1 + agg["avg_overhead"])
        + 0.15 / (1 + agg["avg_hops"])
        + 0.10 * (1 - agg["avg_load_on_path"])
    )
    agg["seed"] = seed
    return agg


def run_ablation(seeds: List[int] = SEEDS, topology: str = TOPOLOGY,
                 num_agents: int = NUM_AGENTS, num_tasks: int = NUM_TASKS,
                 out_dir: str = None):
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "ablation")
    per_seed = []
    for seed in seeds:
        logger.info("=== Ablation SEED %d ===", seed)
        df = run_ablation_seed(seed, topology, num_agents, num_tasks,
                               os.path.join(out_dir, f"seed_{seed}"))
        per_seed.append(df)

    combined = pd.concat(per_seed, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "ablation_all.csv"), index=False)

    numeric = combined.select_dtypes(include=np.number).columns.tolist()
    numeric = [c for c in numeric if c != "seed"]
    mean_df = combined.groupby("method")[numeric].mean()
    std_df  = combined.groupby("method")[numeric].std()
    mean_df.to_csv(os.path.join(out_dir, "ablation_mean.csv"))
    std_df.to_csv(os.path.join(out_dir, "ablation_std.csv"))

    logger.info("\n=== ABLATION MEAN ROUTING SCORE ===\n%s",
                mean_df[["success_rate","avg_effective_latency",
                          "protocol_mismatch_rate","routing_score"]].to_string())
    return mean_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed",  action="store_true")
    p.add_argument("--topology",   default=None)
    p.add_argument("--num_agents", type=int, default=None)
    p.add_argument("--out_dir",    default=None)
    args = p.parse_args()

    import config as cfg
    if args.topology:   cfg.TOPOLOGY   = args.topology
    if args.num_agents: cfg.NUM_AGENTS = args.num_agents

    seeds = SEEDS if args.multiseed else [SEED]
    run_ablation(seeds=seeds, topology=cfg.TOPOLOGY,
                 num_agents=cfg.NUM_AGENTS, num_tasks=cfg.NUM_TASKS,
                 out_dir=args.out_dir)