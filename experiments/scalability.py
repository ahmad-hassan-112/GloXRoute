"""
scalability.py — Scalability experiment for GloXRoute.

Tests routing performance and decision time across increasing network sizes:
  50, 100, 200, 500 nodes

Metrics reported per scale:
  - avg_latency
  - avg_effective_latency
  - avg_hops
  - routing_score
  - avg_decision_time_ms
  - p95_decision_time_ms
  - coordinator build time

Run:
    python scalability.py --seeds 10
"""

import argparse
import logging
import os
import random
import time
from typing import List

import numpy as np
import pandas as pd

from core.config import (
    SEEDS,
    OUTPUT_DIR,
    TOPOLOGY,
    COORDINATOR_UPDATE_INTERVAL
)

from core.network_builder import build_ioa_network
from core.task_generator import generate_tasks
from core.coordinator import GlobalCoordinator
from core.routers import (
    ShortestPathRouter,
    DecentralizedGreedyRouter,
    HybridXAIRouter,
)

from core.metrics import evaluate_route


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

logger = logging.getLogger("scalability")


# Network scales evaluated in the paper
SCALES = [50, 100, 200, 500]

# Number of routing tasks per scale
TASKS_SCALE = 500


def ci95(x):
    """
    95% confidence interval.
    """
    x = np.asarray(x)

    if len(x) <= 1:
        return 0.0

    return (
        1.96 *
        np.std(x, ddof=1) /
        np.sqrt(len(x))
    )


def run_scale(
    num_agents: int,
    seed: int,
    topology: str
) -> List[dict]:

    random.seed(seed)
    np.random.seed(seed)


    G, agents = build_ioa_network(
        seed=seed,
        topology=topology,
        num_agents=num_agents
    )


    tasks = generate_tasks(
        num_tasks=TASKS_SCALE,
        seed=seed,
        num_agents=num_agents
    )


    # Coordinator construction cost
    t0 = time.perf_counter()

    coord = GlobalCoordinator(
        G,
        agents
    )

    coord_build_ms = (
        time.perf_counter() - t0
    ) * 1000.0


    routers = [
        ShortestPathRouter(),

        DecentralizedGreedyRouter(),

        HybridXAIRouter(
            coord,
            router_name="HybridXAI_HeuristicCoord"
        ),
    ]


    records = []


    for router in routers:

        counter = 0

        for task in tasks:

            counter += 1


            if counter % COORDINATOR_UPDATE_INTERVAL == 0:
                coord.update_guidance()


            start = time.perf_counter()


            decision = router.route(
                G,
                agents,
                task
            )


            decision_time = (
                time.perf_counter()
                -
                start
            )


            row = evaluate_route(
                G=G,
                agents=agents,
                task=task,
                method_name=router.name,
                path=decision.path,
                extra_overhead=decision.extra_overhead,
                explanation=decision.explanation or None,
                decision_time=decision_time,
            )


            row["num_agents"] = num_agents
            row["seed"] = seed
            row["coord_build_ms"] = coord_build_ms


            records.append(row)


    return records



def run_scalability(
    scales: List[int] = SCALES,
    seeds: List[int] = None,
    topology: str = TOPOLOGY,
    out_dir: str = None,
):


    if seeds is None:

        # Changed from 3 to 10 seeds
        seeds = SEEDS[:10]


    out_dir = (
        out_dir
        or
        os.path.join(
            OUTPUT_DIR,
            "scalability"
        )
    )


    os.makedirs(
        out_dir,
        exist_ok=True
    )


    all_rows = []


    for n in scales:

        for seed in seeds:

            logger.info(
                "Scale: %d nodes, seed=%d",
                n,
                seed
            )


            rows = run_scale(
                n,
                seed,
                topology
            )


            all_rows.extend(rows)



    df = pd.DataFrame(all_rows)


    df.to_csv(
        os.path.join(
            out_dir,
            "scalability_raw.csv"
        ),
        index=False
    )


    # Exclude tasks solved immediately by source agent
    routing_df = df[
        ~df["source_capable"]
    ].copy()



    agg = routing_df.groupby(
        [
            "method",
            "num_agents"
        ]
    ).agg(

        success_rate_mean=(
            "success",
            "mean"
        ),

        success_rate_std=(
            "success",
            "std"
        ),


        avg_latency_mean=(
            "latency",
            "mean"
        ),

        avg_latency_std=(
            "latency",
            "std"
        ),


        avg_eff_latency_mean=(
            "effective_latency",
            "mean"
        ),

        avg_eff_latency_std=(
            "effective_latency",
            "std"
        ),


        avg_hops_mean=(
            "hops",
            "mean"
        ),

        avg_hops_std=(
            "hops",
            "std"
        ),


        avg_decision_ms_mean=(
            "decision_time_ms",
            "mean"
        ),

        avg_decision_ms_std=(
            "decision_time_ms",
            "std"
        ),


        decision_time_ci95=(
            "decision_time_ms",
            ci95
        ),


        p95_decision_ms=(
            "decision_time_ms",
            lambda x: np.percentile(x,95)
        ),


        coord_build_ms=(
            "coord_build_ms",
            "mean"
        )

    ).reset_index()



    # IoA routing tradeoff score
    agg["routing_score"] = (

        0.30 /
        (
            1 +
            agg["avg_latency_mean"]
        )

        +

        0.20 /
        (
            1 +
            agg["avg_hops_mean"]
        )

        +

        0.10 *
        agg["success_rate_mean"]

    )


    agg.to_csv(
        os.path.join(
            out_dir,
            "scalability_summary.csv"
        ),
        index=False
    )



    logger.info(
        "\n%s",
        agg[
            [
                "method",
                "num_agents",
                "avg_eff_latency_mean",
                "avg_decision_ms_mean",
                "decision_time_ci95",
                "coord_build_ms",
                "routing_score"
            ]
        ].to_string(index=False)
    )


    logger.info(
        "Scalability results saved at %s",
        out_dir
    )


    return agg



if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds"
    )


    parser.add_argument(
        "--topology",
        default=TOPOLOGY
    )


    parser.add_argument(
        "--out_dir",
        default=None
    )


    args = parser.parse_args()



    import core.config as cfg


    cfg.TOPOLOGY = args.topology


    seeds = SEEDS[:args.seeds]


    run_scalability(
        seeds=seeds,
        topology=cfg.TOPOLOGY,
        out_dir=args.out_dir
    )