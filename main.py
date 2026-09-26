"""
main.py — entry point for GloXRoute experiments.

Usage
-----
  # Single seed, BA topology (default)
  python main.py

  # Multi-seed (20 seeds, reports mean ± std + Wilcoxon)
  python main.py --multiseed

  # Congestion scenario, multi-seed
  python main.py --multiseed --congestion

  # Node-failure scenario
  python main.py --multiseed --failure_rate 0.10

  # Topology sweep
  python main.py --multiseed --topology watts_strogatz
"""

import argparse, logging, os, sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed",    action="store_true",
                   help="Run all 20 seeds and report aggregated stats")
    p.add_argument("--topology",     default=None,
                   choices=["barabasi_albert","watts_strogatz","erdos_renyi"])
    p.add_argument("--num_agents",   type=int, default=None)
    p.add_argument("--num_tasks",    type=int, default=None)
    p.add_argument("--seed",         type=int, default=None)
    p.add_argument("--congestion",   action="store_true",
                   help="Enable congestion scenario (30 %% hot agents)")
    p.add_argument("--failure_rate", type=float, default=None,
                   help="Fraction of agents that fail (0.0–1.0)")
    p.add_argument("--out_dir",      default=None)
    return p.parse_args()


def main():
    args = _parse()

    # Override config knobs from CLI
    import core.config as cfg

    if args.topology:      cfg.TOPOLOGY          = args.topology
    if args.num_agents:    cfg.NUM_AGENTS         = args.num_agents
    if args.num_tasks:     cfg.NUM_TASKS          = args.num_tasks
    if args.seed:          cfg.SEED               = args.seed
    if args.congestion:    cfg.CONGESTION_SCENARIO = True
    if args.failure_rate is not None:
        cfg.FAILURE_RATE = args.failure_rate

    tag_parts = [cfg.TOPOLOGY]
    if cfg.CONGESTION_SCENARIO: tag_parts.append("congestion")
    if cfg.FAILURE_RATE > 0:    tag_parts.append(f"fail{cfg.FAILURE_RATE:.2f}")
    tag = "_".join(tag_parts)

    out_dir = args.out_dir or os.path.join(cfg.OUTPUT_DIR, tag)

    if args.multiseed:
        from experiments.experiment import run_multiseed
        logger.info("Multi-seed experiment: topology=%s, n=%d, tasks_per_seed=%d",
                    cfg.TOPOLOGY, cfg.NUM_AGENTS, cfg.NUM_TASKS)
        run_multiseed(
            seeds      = cfg.SEEDS,
            topology   = cfg.TOPOLOGY,
            num_agents = cfg.NUM_AGENTS,
            num_tasks  = cfg.NUM_TASKS,
            out_dir    = out_dir,
        )
    else:
        from experiments.experiment import GloXRouteExperiment
        seed = args.seed or cfg.SEED
        logger.info("Single-seed experiment: seed=%d topology=%s", seed, cfg.TOPOLOGY)
        exp = GloXRouteExperiment(
            seed       = seed,
            topology   = cfg.TOPOLOGY,
            num_agents = cfg.NUM_AGENTS,
            num_tasks  = cfg.NUM_TASKS,
            run_dir    = out_dir,
        )
        exp.run()


if __name__ == "__main__":
    main()