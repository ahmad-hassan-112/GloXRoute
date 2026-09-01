"""
experiment.py — GloXRoute evaluation framework.

Key changes vs original
------------------------
1.  Composite metric completely rewritten:
      - NO explainability bonus in routing comparison
      - Individual metrics reported separately
      - Jain fairness index added for load balance
2.  Overhead: HybridXAI coordinator precomputation cost is now
    charged (amortised over NUM_TASKS) and reported transparently.
3.  Source-capable tasks excluded from routing metrics (denominator fix).
4.  Multi-seed runner added (run_multiseed).
5.  Neural / GNN variants removed from main comparison; kept only as
    an optional ablation flag.
"""

import os, time, json, logging, random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
from scipy import stats as scipy_stats

from config import (
    SEED, SEEDS, NUM_TASKS, OUTPUT_DIR, COORDINATOR_UPDATE_INTERVAL,
    TOPOLOGY, NUM_AGENTS,
)
from models import Agent, Task
from network_builder import build_ioa_network
from task_generator import generate_tasks
from coordinator import GlobalCoordinator, LearnedGlobalCoordinator
from routers import (
    ShortestPathRouter,
    CentralizedRouter,
    DecentralizedGreedyRouter,
    BiRouterStyleRouter,
    HybridXAIRouter,
)
from metrics import evaluate_route

logger = logging.getLogger("experiment")


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

def _jain_fairness(loads: List[float]) -> float:
    """Jain's fairness index for load distribution on path nodes."""
    if not loads:
        return 1.0
    n = len(loads)
    s  = sum(loads)
    s2 = sum(x**2 for x in loads)
    return (s**2) / (n * s2) if s2 > 0 else 1.0


def _coordinator_overhead_per_task(
    G: nx.Graph,
    coordinator: GlobalCoordinator,
    num_tasks: int,
) -> float:
    """
    Amortised control-message cost of the global coordinator.
    We count one "state-advertisement" message per node per capability
    per update round, amortised over the task batch.
    This is conservative (a lower bound on the real cost).
    """
    n_nodes = len(G.nodes())
    from config import CAPABILITIES
    n_caps  = len(CAPABILITIES)
    update_rounds = max(1, num_tasks // COORDINATOR_UPDATE_INTERVAL)
    total_msgs = n_nodes * n_caps * update_rounds
    return total_msgs / max(1, num_tasks)   # msgs per task


# ================================================================== #
#  Single-seed experiment                                             #
# ================================================================== #

class GloXRouteExperiment:

    def __init__(self, seed: int = SEED, topology: str = TOPOLOGY,
                 num_agents: int = NUM_AGENTS, num_tasks: int = NUM_TASKS,
                 run_dir: Optional[str] = None):
        self.seed       = seed
        self.topology   = topology
        self.num_agents = num_agents
        self.num_tasks  = num_tasks

        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = run_dir or os.path.join(OUTPUT_DIR, ts)
        os.makedirs(self.run_dir, exist_ok=True)

        # Seeding
        random.seed(seed)
        np.random.seed(seed)

        # Network
        logger.info("Building network (topology=%s, n=%d, seed=%d)",
                    topology, num_agents, seed)
        self.G, self.agents = build_ioa_network(
            seed=seed, topology=topology, num_agents=num_agents
        )

        # Tasks
        all_tasks = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
        self.test_tasks = all_tasks   # all tasks used for testing

        # Coordinator
        t0 = time.perf_counter()
        self.coordinator = GlobalCoordinator(self.G, self.agents)
        self.coord_build_time = time.perf_counter() - t0

        # Amortised coordinator overhead (control msgs / task)
        self.coord_overhead_per_task = _coordinator_overhead_per_task(
            self.G, self.coordinator, num_tasks
        )

        # Routers (NO neural/GNN in main comparison)
        self.routers = [
            ShortestPathRouter(),
            CentralizedRouter(),
            DecentralizedGreedyRouter(),
            BiRouterStyleRouter(),
            HybridXAIRouter(self.coordinator, router_name="HybridXAI_HeuristicCoord"),
        ]

    # -------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        all_rows = []

        for router in self.routers:
            logger.info("  Running %s ...", router.name)
            update_counter = 0

            for task in self.test_tasks:
                update_counter += 1
                if update_counter % COORDINATOR_UPDATE_INTERVAL == 0:
                    if hasattr(self.coordinator, "update_guidance"):
                        self.coordinator.update_guidance()

                t0   = time.perf_counter()
                dec  = router.route(self.G, self.agents, task)
                dt   = time.perf_counter() - t0

                # Overhead: HybridXAI gets its coordinator cost added
                extra = dec.extra_overhead
                if "HybridXAI" in router.name:
                    extra += self.coord_overhead_per_task

                row = evaluate_route(
                    G            = self.G,
                    agents       = self.agents,
                    task         = task,
                    method_name  = router.name,
                    path         = dec.path,
                    extra_overhead = extra,
                    explanation  = dec.explanation or None,
                    decision_time = dt,
                )
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(self.run_dir, "all_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Raw results saved to %s", csv_path)

        summary = self._summarise(df)
        sum_path = os.path.join(self.run_dir, "summary.csv")
        summary.to_csv(sum_path)
        logger.info("Summary:\n%s", summary.to_string())
        return summary

    # -------------------------------------------------------------- #
    def _summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-method summary statistics.
        Routing tasks only (source_capable == False).
        Explainability reported separately — NOT included in routing score.
        """
        routing_df = df[~df["source_capable"]].copy()

        agg = routing_df.groupby("method").agg(
            success_rate          = ("success", "mean"),
            avg_latency           = ("latency", "mean"),
            median_latency        = ("latency", "median"),
            p95_latency           = ("latency", lambda x: np.percentile(x, 95)),
            avg_effective_latency = ("effective_latency", "mean"),
            p95_effective_latency = ("effective_latency", lambda x: np.percentile(x, 95)),
            latency_std           = ("latency", "std"),
            avg_hops              = ("hops", "mean"),
            avg_overhead          = ("communication_overhead", "mean"),
            avg_path_stretch      = ("path_stretch", lambda x: x.dropna().mean()),
            loop_rate             = ("loop_detected", "mean"),
            avg_load_on_path      = ("avg_load_on_path", "mean"),
            max_load_on_path      = ("max_load_on_path", "mean"),
            protocol_mismatch_rate= ("protocol_mismatch_rate", "mean"),
            avg_decision_time_ms  = ("decision_time_ms", "mean"),
            p95_decision_time_ms  = ("decision_time_ms", lambda x: np.percentile(x, 95)),
        ).reset_index()

        # Explainability — computed over ALL tasks (including source-capable)
        xai = df.groupby("method").agg(
            explanation_rate   = ("explanation_available", "mean"),
            avg_explanation_len = ("explanation_length", "mean"),
        ).reset_index()
        agg = agg.merge(xai, on="method")

        # Source-capable fraction
        src = df.groupby("method").agg(
            source_capable_rate = ("source_capable", "mean"),
        ).reset_index()
        agg = agg.merge(src, on="method")

        # Jain fairness on load
        def jain(x):
            loads = list(x.dropna())
            return _jain_fairness(loads)

        jf = routing_df.groupby("method").agg(
            load_jain_fairness = ("avg_load_on_path", jain),
        ).reset_index()
        agg = agg.merge(jf, on="method")

        # ---- Routing score (NO explainability bonus) ----
        # Individual scores for each dimension
        agg["latency_score"]  = 1.0 / (1.0 + agg["avg_latency"])
        agg["overhead_score"] = 1.0 / (1.0 + agg["avg_overhead"])
        agg["hop_score"]      = 1.0 / (1.0 + agg["avg_hops"])
        agg["stretch_score"]  = 1.0 / (1.0 + agg["avg_path_stretch"])
        agg["load_score"]     = 1.0 - agg["avg_load_on_path"]   # lower load = better

        # Composite routing score (pure routing, no XAI bonus)
        # Weights: latency 30%, eff-latency via stretch 25%,
        #          overhead 20%, hops 15%, load balance 10%
        agg["routing_score"] = (
            0.30 * agg["latency_score"]
            + 0.25 * agg["stretch_score"]
            + 0.20 * agg["overhead_score"]
            + 0.15 * agg["hop_score"]
            + 0.10 * agg["load_score"]
        )

        # Explainability score — reported separately, never added to routing_score
        agg["explainability_score"] = agg["explanation_rate"]

        return agg.set_index("method")


# ================================================================== #
#  Multi-seed runner                                                  #
# ================================================================== #

def run_multiseed(
    seeds: List[int] = SEEDS,
    topology: str = TOPOLOGY,
    num_agents: int = NUM_AGENTS,
    num_tasks: int = NUM_TASKS,
    out_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Run full experiment across multiple seeds and return a combined
    summary with mean ± std and Wilcoxon p-values vs ShortestPath.
    """
    os.makedirs(out_dir, exist_ok=True)
    per_seed: List[pd.DataFrame] = []

    for seed in seeds:
        logger.info("=== SEED %d ===", seed)
        seed_dir = os.path.join(out_dir, f"seed_{seed}")
        exp = GloXRouteExperiment(
            seed=seed, topology=topology,
            num_agents=num_agents, num_tasks=num_tasks,
            run_dir=seed_dir,
        )
        summary = exp.run()
        summary["seed"] = seed
        per_seed.append(summary.reset_index())

    combined = pd.concat(per_seed, ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "multiseed_all.csv"), index=False)

    # Aggregate: mean ± std across seeds
    numeric_cols = combined.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "seed"]

    agg_mean = combined.groupby("method")[numeric_cols].mean()
    agg_std  = combined.groupby("method")[numeric_cols].std()

    # Wilcoxon signed-rank test: each method vs ShortestPath on routing_score
    wilcoxon_p = {}
    sp_scores = combined[combined["method"] == "ShortestPath"]["routing_score"].values
    for method in combined["method"].unique():
        if method == "ShortestPath":
            wilcoxon_p[method] = np.nan
            continue
        m_scores = combined[combined["method"] == method]["routing_score"].values
        if len(m_scores) < 5:
            wilcoxon_p[method] = np.nan
            continue
        try:
            _, p = scipy_stats.wilcoxon(m_scores, sp_scores, alternative="greater")
        except Exception:
            p = np.nan
        wilcoxon_p[method] = p

    agg_mean["wilcoxon_p_vs_sp"] = pd.Series(wilcoxon_p)

    # Cohen's d vs ShortestPath
    sp_mean = agg_mean.loc["ShortestPath", "routing_score"] if "ShortestPath" in agg_mean.index else np.nan
    sp_std  = agg_std.loc["ShortestPath", "routing_score"]  if "ShortestPath" in agg_std.index  else np.nan

    def cohens_d(m_mean, m_std):
        pooled = np.sqrt((m_std**2 + sp_std**2) / 2)
        return (m_mean - sp_mean) / pooled if pooled > 0 else 0.0

    agg_mean["cohens_d_routing_vs_sp"] = agg_mean["routing_score"].apply(
        lambda m: cohens_d(m, agg_std.loc[agg_mean[agg_mean["routing_score"]==m].index[0], "routing_score"])
        if not np.isnan(m) else np.nan
    )

    # Save
    agg_mean.to_csv(os.path.join(out_dir, "multiseed_mean.csv"))
    agg_std.to_csv(os.path.join(out_dir, "multiseed_std.csv"))

    logger.info("\n=== MULTI-SEED MEAN ===\n%s", agg_mean[
        ["success_rate","avg_latency","avg_effective_latency",
         "routing_score","explainability_score","wilcoxon_p_vs_sp"]
    ].to_string())

    return agg_mean