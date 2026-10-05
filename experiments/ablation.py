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

from core.config import TIE_BREAK_TOP_K
from core.config import (
    SEED, SEEDS, NUM_TASKS, OUTPUT_DIR, COORDINATOR_UPDATE_INTERVAL,
    TOPOLOGY, NUM_AGENTS, LATENCY_BOUND_EPSILON, ROUTE_CACHE_MAXSIZE, MAX_HOPS,
)
from core.models import Agent, Task, RouteDecision
from core.network_builder import build_ioa_network
from core.task_generator import generate_tasks
from core.coordinator import GlobalCoordinator
from core.metrics import evaluate_route, set_jitter_seed
from core.routers import HybridXAIRouter
from core.dynamics import LoadDynamics
from core.utils import capable_nodes, source_can_solve, effective_edge_weight
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ablation")


# ================================================================== #
#  Configurable HybridXAI for ablation                                #
# ================================================================== #

def AblatedHybridXAIRouter(coordinator: GlobalCoordinator, name: str, **switches):
    """The ablation uses the SAME router as the main experiment
    (core.routers.HybridXAIRouter), with individual components switched off."""
    return HybridXAIRouter(coordinator, router_name=name, **switches)


# ================================================================== #
#  Ablation runner                                                     #
# ================================================================== #

ABLATION_VARIANTS = [
    dict(name="GloXRoute_Full",         use_global=True,  use_capability=True,  use_load=True,  use_reliability=True,  use_protocol=True),
    dict(name="Ablation_NoDeadlineBudget", deadline_aware=False),
    dict(name="Ablation_NoCostTerm",       use_cost=False),
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

    set_jitter_seed(seed)
    G, agents = build_ioa_network(seed=seed, topology=topology, num_agents=num_agents)
    tasks     = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
    coord     = GlobalCoordinator(G, agents)
    dynamics  = LoadDynamics(agents, seed=seed)

    routers = [AblatedHybridXAIRouter(coord, **v) for v in ABLATION_VARIANTS]

    all_rows = []
    for router in routers:
        logger.info("  Ablation variant: %s", router.name)
        counter = 0
        dynamics.reset()
        coord.update_guidance()
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
            dynamics.step()

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
        deadline_miss_rate    = ("deadline_miss", "mean"),
    ).reset_index()

    # overhead reported separately (as in the main experiment); remaining
    # weights keep their ratios
    agg["routing_score"] = (
        0.30 / (1 + agg["avg_latency"])
        + 0.25 / (1 + agg["avg_path_stretch"])
        + 0.15 / (1 + agg["avg_hops"])
        + 0.10 * (1 - agg["avg_load_on_path"])
    ) / 0.80
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
                mean_df[["success_rate","avg_effective_latency","deadline_miss_rate",
                          "protocol_mismatch_rate","routing_score"]].to_string())
    return mean_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed",  action="store_true")
    p.add_argument("--topology",   default=None)
    p.add_argument("--num_agents", type=int, default=None)
    p.add_argument("--dynamic",    action="store_true")
    p.add_argument("--out_dir",    default=None)
    args = p.parse_args()

    import core.config as cfg
    if args.topology:   cfg.TOPOLOGY   = args.topology
    if args.num_agents: cfg.NUM_AGENTS = args.num_agents
    if args.dynamic:    cfg.DYNAMIC_LOADS = True
    out_dir = args.out_dir or os.path.join(
        cfg.OUTPUT_DIR, "ablation_dynamic" if args.dynamic else "ablation")

    seeds = SEEDS if args.multiseed else [SEED]
    run_ablation(seeds=seeds, topology=cfg.TOPOLOGY,
                 num_agents=cfg.NUM_AGENTS, num_tasks=cfg.NUM_TASKS,
                 out_dir=out_dir)