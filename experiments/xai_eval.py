"""
xai_eval.py — evaluate GloXRoute's explanations on the ROUTER'S OWN decisions.

The router (core.routers.HybridXAIRouter) runs with record_trace=True, so
every hop's full candidate table, the feasible set and the chosen neighbour
are recorded exactly as the router saw them. Nothing is re-implemented here.

  1. Fidelity: does the emitted explanation reproduce the decision?
       (a) the feature values printed in the explanation, multiplied by the
           published weights, give the printed tie_score, and
       (b) the chosen neighbour has the highest tie_score in the feasible set.
     A decision passes only if both hold.
  2. Counterfactual robustness: for decisions with a feasible runner-up, the
     load increase on the winner that would flip the choice. The load feature
     enters the score twice (load and congestion terms), so
         delta_load_to_flip = score_gap / (|w_load| + |w_congestion|).
     This is a score-level counterfactual: it does not re-check the deadline
     budget after the change.
  3. Feature contribution stability: weight x value of every winning
     candidate, per feature (mean and SD over all decisions).

Outputs (unchanged file names): xai_summary.csv, counterfactuals.csv,
feature_consistency.csv, plus xai_decisions.csv (one row per decision).
"""
import argparse, logging, os, random, re
import numpy as np
import pandas as pd

from core.config import SEED, NUM_AGENTS, NUM_TASKS, OUTPUT_DIR, TOPOLOGY, COORDINATOR_UPDATE_INTERVAL
from core.network_builder import build_ioa_network
from core.task_generator import generate_tasks
from core.coordinator import GlobalCoordinator
from core.routers import HybridXAIRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("xai_eval")

W = HybridXAIRouter.SCORE_WEIGHTS
LOAD_SENSITIVITY = abs(W["load"]) + abs(W["congestion"])
# explanations print 5 decimals: worst-case rounding error of the recomputed score
REPRO_TOL = 0.5e-5 * (sum(abs(w) for w in W.values()) + 1) + 1e-9


def _parse_explanation(expl: str) -> dict:
    return {k: float(v) for k, v in re.findall(r"(\w+)=(-?\d+\.\d+)", expl)}


def collect_trace(G, agents, coordinator, tasks):
    router = HybridXAIRouter(coordinator, router_name="GloXRoute", record_trace=True)
    for i, task in enumerate(tasks):
        if i % COORDINATOR_UPDATE_INTERVAL == 0:
            coordinator.update_guidance()
        router.route(G, agents, task)
    # keep scored decisions (fallback hops have no candidate table)
    return [h for h in router.trace if h["rows"]]


def evaluate_fidelity(trace):
    ok_repro = ok_argmax = both = 0
    rows = []
    for h in trace:
        vals = _parse_explanation(h["explanation"])
        recomputed = sum(W[f] * vals[f] for f in W)
        repro = abs(recomputed - vals["tie_score"]) <= REPRO_TOL
        feas = [r for r in h["rows"] if r["node"] in h["feasible"]]
        argmax = h["chosen"] == max(feas, key=lambda r: r["tie_score"])["node"]
        ok_repro += repro; ok_argmax += argmax; both += (repro and argmax)
        rows.append({"task_id": h["task_id"], "current": h["current"], "chosen": h["chosen"],
                     "n_candidates": len(h["rows"]), "n_feasible": len(feas),
                     "explanation_reproduces_score": repro, "chosen_is_argmax": argmax})
    n = max(1, len(trace))
    logger.info("Fidelity: %.4f (explanation reproduces score %.4f, argmax %.4f) over %d decisions",
                both / n, ok_repro / n, ok_argmax / n, len(trace))
    return {"fidelity": both / n, "score_reproduction": ok_repro / n,
            "argmax_consistency": ok_argmax / n, "decisions_checked": len(trace)}, pd.DataFrame(rows)


def evaluate_counterfactuals(trace):
    recs = []
    for h in trace:
        feas = sorted([r for r in h["rows"] if r["node"] in h["feasible"]],
                      key=lambda r: -r["tie_score"])
        if len(feas) < 2:
            continue
        gap = feas[0]["tie_score"] - feas[1]["tie_score"]
        recs.append({"task_id": h["task_id"], "current": h["current"],
                     "winner": feas[0]["node"], "runner_up": feas[1]["node"],
                     "score_gap": gap, "delta_load_to_flip": gap / LOAD_SENSITIVITY})
    df = pd.DataFrame(recs)
    if not df.empty:
        logger.info("Counterfactuals (%d decisions with a feasible runner-up): median=%.3f mean=%.3f min=%.3f",
                    len(df), df.delta_load_to_flip.median(), df.delta_load_to_flip.mean(),
                    df.delta_load_to_flip.min())
    return df


def evaluate_feature_consistency(trace):
    contrib = {f: [] for f in W}
    for h in trace:
        win = next(r for r in h["rows"] if r["node"] == h["chosen"])
        for f in W:
            contrib[f].append(W[f] * win[f])
    df = pd.DataFrame([{"feature": f, "weight": W[f],
                        "mean_contribution": float(np.mean(v)) if v else 0.0,
                        "std_contribution": float(np.std(v)) if v else 0.0,
                        "n_decisions": len(v)} for f, v in contrib.items()])
    return df.sort_values("mean_contribution", ascending=False)


def run_xai_eval(seed: int = SEED, num_tasks: int = NUM_TASKS, topology: str = TOPOLOGY,
                 num_agents: int = NUM_AGENTS, out_dir: str = None):
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "xai_eval")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed); np.random.seed(seed)

    G, agents = build_ioa_network(seed=seed, topology=topology, num_agents=num_agents)
    tasks     = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
    coord     = GlobalCoordinator(G, agents)

    trace = collect_trace(G, agents, coord, tasks)
    fid, dec_df = evaluate_fidelity(trace)
    cf_df = evaluate_counterfactuals(trace)
    fc_df = evaluate_feature_consistency(trace)

    dec_df.to_csv(os.path.join(out_dir, "xai_decisions.csv"), index=False)
    cf_df.to_csv(os.path.join(out_dir, "counterfactuals.csv"), index=False)
    fc_df.to_csv(os.path.join(out_dir, "feature_consistency.csv"), index=False)

    summary = {
        **fid,
        "decisions_with_runner_up":   len(cf_df),
        "counterfactual_median_delta": cf_df["delta_load_to_flip"].median() if not cf_df.empty else None,
        "counterfactual_mean_delta":   cf_df["delta_load_to_flip"].mean()   if not cf_df.empty else None,
        "counterfactual_min_delta":    cf_df["delta_load_to_flip"].min()    if not cf_df.empty else None,
        "top_feature":                 fc_df.iloc[0]["feature"],
        "top_feature_mean_contrib":    fc_df.iloc[0]["mean_contribution"],
    }
    pd.Series(summary).to_csv(os.path.join(out_dir, "xai_summary.csv"))
    logger.info("\nXAI Summary:\n%s\n\nFeature contributions:\n%s",
                pd.Series(summary).to_string(), fc_df.to_string(index=False))
    return summary, cf_df, fc_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=SEED)
    p.add_argument("--num_tasks",  type=int, default=NUM_TASKS)
    p.add_argument("--topology",   default=TOPOLOGY)
    p.add_argument("--num_agents", type=int, default=NUM_AGENTS)
    p.add_argument("--out_dir",    default=None)
    a = p.parse_args()
    run_xai_eval(seed=a.seed, num_tasks=a.num_tasks, topology=a.topology,
                 num_agents=a.num_agents, out_dir=a.out_dir)
