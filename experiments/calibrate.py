"""
calibrate.py — choose the weight of GloXRoute's cost_advantage score term.

The weight is selected on TRAINING graphs only: seeds DRLOR_TRAIN_SEED_OFFSET
.. +DRLOR_TRAIN_GRAPHS-1 (the same held-out graphs DRL-OR-S trains on), which
never appear in config.SEEDS. The chosen value is the smallest weight whose
composite score is within 0.0005 of the best (the start of the plateau), so
the other interpretable terms still break ties.

Usage:  python3 -m experiments.calibrate [--topology barabasi_albert]
Writes: <OUTPUT_DIR>/calibration/cost_weight_grid.csv
Result used in the paper: 50 (plateau from 50; see the CSV).
"""
import argparse, logging, os
import numpy as np
import pandas as pd

import core.config as cfg
from core.routers import HybridXAIRouter
import experiments.experiment as E

logging.basicConfig(level=logging.WARNING)
GRID = [0, 2, 5, 10, 20, 50, 100, 200, 1000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default=cfg.TOPOLOGY)
    a = p.parse_args()
    seeds = [cfg.DRLOR_TRAIN_SEED_OFFSET + i for i in range(cfg.DRLOR_TRAIN_GRAPHS)]
    assert not set(seeds) & set(cfg.SEEDS), "training seeds overlap evaluation seeds"
    out = os.path.join(cfg.OUTPUT_DIR, "calibration")
    rows = []
    for seed in seeds:
        e = E.GloXRouteExperiment(seed=seed, topology=a.topology, num_agents=cfg.NUM_AGENTS,
                                  num_tasks=cfg.NUM_TASKS, run_dir=os.path.join(out, f"seed_{seed}"))
        routers = []
        for w in GRID:
            r = HybridXAIRouter(e.coordinator, router_name=f"w={w}")
            r.SCORE_WEIGHTS = dict(HybridXAIRouter.SCORE_WEIGHTS, cost_advantage=float(w))
            routers.append(r)
        e.routers = routers
        s = e.run()
        for w in GRID:
            rows.append({"seed": seed, "cost_weight": w, **s.loc[f"w={w}", [
                "routing_score", "deadline_miss_rate", "avg_effective_latency",
                "protocol_mismatch_rate"]].to_dict()})
    df = pd.DataFrame(rows)
    grid = df.groupby("cost_weight").mean(numeric_only=True).drop(columns="seed")
    best = grid["routing_score"].max()
    chosen = grid.index[grid["routing_score"] >= best - 5e-4].min()
    df.to_csv(os.path.join(out, "cost_weight_grid.csv"), index=False)
    print(grid.round(4).to_string())
    print(f"\nChosen cost_advantage weight: {chosen}  (set in core/routers.py SCORE_WEIGHTS)")


if __name__ == "__main__":
    main()
