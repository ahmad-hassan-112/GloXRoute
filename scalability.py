"""
scalability.py — Scalability experiment for GloXRoute.

Tests routing performance and decision time across increasing network sizes:
  50, 100, 200, 500 nodes (configurable via SCALES)

Metrics reported per scale:
  - avg_latency, avg_effective_latency
  - avg_hops, routing_score
  - avg_decision_time_ms, p95_decision_time_ms
  - coordinator build time (ms)

Run with:
  python scalability.py
  python scalability.py --seeds 3
"""

import argparse, logging, os, random, time
import numpy as np
import pandas as pd
from typing import List

from config import SEED, SEEDS, OUTPUT_DIR, TOPOLOGY, NUM_TASKS, COORDINATOR_UPDATE_INTERVAL
from network_builder import build_ioa_network
from task_generator import generate_tasks
from coordinator import GlobalCoordinator
from routers import ShortestPathRouter, DecentralizedGreedyRouter, HybridXAIRouter
from metrics import evaluate_route

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("scalability")

SCALES      = [50, 100, 200, 500]
TASKS_SCALE = 500   # tasks per scale point (keep small for speed)


def run_scale(num_agents: int, seed: int, topology: str) -> List[dict]:
    random.seed(seed)
    np.random.seed(seed)

    G, agents = build_ioa_network(seed=seed, topology=topology, num_agents=num_agents)
    tasks = generate_tasks(num_tasks=TASKS_SCALE, seed=seed, num_agents=num_agents)

    t0    = time.perf_counter()
    coord = GlobalCoordinator(G, agents)
    coord_build_ms = (time.perf_counter() - t0) * 1000.0

    routers = [
        ShortestPathRouter(),
        DecentralizedGreedyRouter(),
        HybridXAIRouter(coord, router_name="HybridXAI_HeuristicCoord"),
    ]

    records = []
    for router in routers:
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
            row["num_agents"]       = num_agents
            row["seed"]             = seed
            row["coord_build_ms"]   = coord_build_ms
            records.append(row)

    return records


def run_scalability(
    scales: List[int] = SCALES,
    seeds: List[int]  = None,
    topology: str     = TOPOLOGY,
    out_dir: str      = None,
):
    if seeds is None:
        seeds = SEEDS[:3]   # default: 3 seeds for speed
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "scalability")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for n in scales:
        for seed in seeds:
            logger.info("Scale: %d nodes, seed=%d", n, seed)
            rows = run_scale(n, seed, topology)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "scalability_raw.csv"), index=False)

    routing_df = df[~df["source_capable"]].copy()
    agg = routing_df.groupby(["method", "num_agents"]).agg(
        success_rate         = ("success", "mean"),
        avg_latency          = ("latency", "mean"),
        avg_eff_latency      = ("effective_latency", "mean"),
        avg_hops             = ("hops", "mean"),
        avg_decision_ms      = ("decision_time_ms", "mean"),
        p95_decision_ms      = ("decision_time_ms", lambda x: np.percentile(x, 95)),
        coord_build_ms       = ("coord_build_ms", "mean"),
    ).reset_index()

    agg["routing_score"] = (
        0.30 / (1 + agg["avg_latency"])
        + 0.20 / (1 + agg["avg_hops"])
        + 0.10 * agg["success_rate"]
    )

    agg.to_csv(os.path.join(out_dir, "scalability_summary.csv"), index=False)
    logger.info("\n%s", agg[["method","num_agents","avg_eff_latency",
                              "avg_decision_ms","routing_score"]].to_string(index=False))
    return agg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",    type=int, default=3, help="Number of seeds to use")
    p.add_argument("--topology", default=TOPOLOGY)
    p.add_argument("--out_dir",  default=None)
    args = p.parse_args()

    import config as cfg
    if args.topology: cfg.TOPOLOGY = args.topology
    seeds = SEEDS[:args.seeds]
    run_scalability(seeds=seeds, topology=cfg.TOPOLOGY, out_dir=args.out_dir)