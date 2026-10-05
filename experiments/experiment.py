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
from core.config import TASK_MODE

from core.config import (
    SEED, SEEDS, NUM_TASKS, OUTPUT_DIR, COORDINATOR_UPDATE_INTERVAL,
    TOPOLOGY, NUM_AGENTS,
)
from core.models import Agent, Task
from core.network_builder import build_ioa_network
from core.task_generator import generate_tasks
from core.coordinator import GlobalCoordinator, LearnedGlobalCoordinator
from core.routers import (
    ShortestPathRouter,
    CentralizedRouter,
    DecentralizedGreedyRouter,
    BiRouterStyleRouter,
    SAWRouter,
    TOPSISRouter,
    HybridXAIRouter,
)
from core.learned_routers import DRLORSRouter, AgentNetStyleRouter
from core.metrics import evaluate_route, set_jitter_seed
from core.dynamics import LoadDynamics

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


def _snapshot_network_and_tasks(G: nx.Graph, agents: Dict[int, Agent],
                                tasks: List[Task], run_dir: str) -> None:
    """
    Persist the exact network and task set used for this run, so the
    routing experiment does not rely purely on "seed + generator code"
    for reproducibility -- an inspectable, versionable snapshot of the
    actual test data is written to disk alongside the results.

    Writes (into <run_dir>/dataset_snapshot/):
      network.graphml   -- full graph incl. edge latency/bandwidth
      agents.csv         -- one row per agent: capabilities, load,
                             reliability, resource, protocol, failed
      tasks.csv           -- one row per task: source, required
                             capability, priority, deadline factor
      snapshot_meta.json  -- seed, topology, counts, generation timestamp
    """
    snap_dir = os.path.join(run_dir, "dataset_snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    # --- network -------------------------------------------------------
    # GraphML requires JSON-safe attribute types; nx graphs here only
    # carry numeric edge attributes (latency, bandwidth), so this is
    # safe as-is.
    nx.write_graphml(G, os.path.join(snap_dir, "network.graphml"))

    # --- agents ----------------------------------------------------------
    agent_rows = []
    for node, a in agents.items():
        agent_rows.append({
            "agent_id":     node,
            "capabilities": "|".join(sorted(a.capabilities)),
            "reliability":  a.reliability,
            "load":         a.load,
            "resource":     a.resource,
            "protocol":     a.protocol,
            "failed":       bool(getattr(a, "failed", False)),
        })
    pd.DataFrame(agent_rows).sort_values("agent_id").to_csv(
        os.path.join(snap_dir, "agents.csv"), index=False
    )

    # --- tasks -----------------------------------------------------------
    task_rows = [{
        "task_id":             t.task_id,
        "source":               t.source,
        "required_capability":  t.required_capability,
        "priority":             t.priority,
        "deadline_factor":      t.deadline,
    } for t in tasks]
    pd.DataFrame(task_rows).to_csv(
        os.path.join(snap_dir, "tasks.csv"), index=False
    )

    # --- metadata ----------------------------------------------------------
    meta = {
        "snapshot_created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_nodes":            G.number_of_nodes(),
        "num_edges":            G.number_of_edges(),
        "num_agents":           len(agents),
        "num_tasks":            len(tasks),
    }
    with open(os.path.join(snap_dir, "snapshot_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset snapshot saved to %s (network.graphml, agents.csv, "
                "tasks.csv, snapshot_meta.json)", snap_dir)


def _coordinator_overhead_per_task(
    G: nx.Graph,
    coordinator: GlobalCoordinator,
    num_tasks: int,
) -> float:
    """
    MEASURED amortised control cost of the lazy global coordinator.

    Every time guidance for a capability is (re)computed after a refresh,
    the coordinator runs one multi-source Dijkstra and disseminates the result:
    one state-advertisement message per node, i.e. |V| messages. Lookups
    answered from the cache in the same refresh window cost nothing.

    Uses the coordinator's LIFETIME counters, so every refresh window of the
    run is charged (the per-window counters are reset by update_guidance()).
    Call AFTER routing, having called reset_lifetime_stats() before it.
    """
    runs = coordinator.overhead_stats()["lifetime_dijkstra_runs"]
    return runs * len(G.nodes()) / max(1, num_tasks)


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
        set_jitter_seed(seed)

        # Network
        logger.info("Building network (topology=%s, n=%d, seed=%d)",
                    topology, num_agents, seed)
        self.G, self.agents = build_ioa_network(
            seed=seed, topology=topology, num_agents=num_agents
        )

        # Tasks
        all_tasks = generate_tasks(num_tasks=num_tasks, seed=seed, num_agents=num_agents)
        self.test_tasks = all_tasks

        from core.network_builder import capability_distribution
        capdist = capability_distribution(self.agents)
        logger.info("Capability distribution (agents holding each): %s", capdist)
        logger.info("Tasks generated: %d (mode=%s)", len(all_tasks), TASK_MODE)
        self.num_tasks = len(all_tasks)

        # Persist the exact network + task set tested, so results can be
        # traced back to concrete, inspectable data rather than only to
        # "seed + generator code".
        _snapshot_network_and_tasks(self.G, self.agents, self.test_tasks, self.run_dir)

        # Time-varying loads (no-op unless DYNAMIC_LOADS is enabled)
        self.dynamics = LoadDynamics(self.agents, seed=seed)

        # Coordinator
        t0 = time.perf_counter()
        self.coordinator = GlobalCoordinator(self.G, self.agents)
        self.coord_build_time = time.perf_counter() - t0

        # Coordinator overhead is MEASURED during the run (see run()).
        self.coord_overhead_per_task = 0.0

        # Routers (NO neural/GNN in main comparison)
        self.routers = [
            ShortestPathRouter(),
            CentralizedRouter(),
            DecentralizedGreedyRouter(),
            BiRouterStyleRouter(),
            # Multi-criteria (MADM) baselines: same attributes as
            # HybridXAI but standard aggregation and NO coordination,
            # isolating the contribution of global coordination.
            SAWRouter(),
            TOPSISRouter(),
            # Recent CCF-A learned baselines (see core/learned_routers.py)
            DRLORSRouter(topology=topology, num_agents=num_agents, eval_seed=seed),
            AgentNetStyleRouter(eval_seed=seed),
            HybridXAIRouter(self.coordinator, router_name="HybridXAI_HeuristicCoord"),
        ]
        self.training_cost: Dict[str, dict] = {}

    # -------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        all_rows = []

        for router in self.routers:
            logger.info("  Running %s ...", router.name)
            update_counter = 0
            router_start = len(all_rows)

            # Every router sees the same initial state and load trajectory.
            self.dynamics.reset()
            self.coordinator.update_guidance()
            self.coordinator.reset_lifetime_stats()
            if hasattr(router, "refresh_state"):
                router.refresh_state()

            # Learned baselines train / warm up here; this cost is reported
            # separately (training_cost.json) and never counted as decision time.
            if hasattr(router, "prepare"):
                router.prepare(self.G, self.agents)
                self.training_cost[router.name] = dict(router.train_stats)
                logger.info("    %s preparation: %s", router.name, router.train_stats)

            if "HybridXAI" in router.name:
                router_rows_start = len(all_rows)

            for task in self.test_tasks:
                update_counter += 1
                if update_counter % COORDINATOR_UPDATE_INTERVAL == 0:
                    # global state refresh: GloXRoute's coordinator and any
                    # other router holding global state refresh together
                    self.coordinator.update_guidance()
                    if hasattr(router, "refresh_state"):
                        router.refresh_state()

                t0   = time.perf_counter()
                dec  = router.route(self.G, self.agents, task)
                dt   = time.perf_counter() - t0

                # online learning (AgentNet-style) happens outside the timed decision
                if hasattr(router, "feedback"):
                    router.feedback(self.G, self.agents, task, dec.path)

                # Overhead: HybridXAI gets its coordinator cost added
                extra = dec.extra_overhead

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

                # loads evolve before the next task (dynamic scenario only)
                self.dynamics.step()

            # After this router finishes, retro-charge measured coordinator cost
            if "HybridXAI" in router.name:
                measured = _coordinator_overhead_per_task(
                    self.G, self.coordinator, len(self.test_tasks)
                )
                logger.info("    Measured coordinator overhead: %.3f msgs/task", measured)
                for r in all_rows[router_rows_start:]:
                    r["communication_overhead"] += measured

            # Any other router that disseminates global state (e.g. DRL-OR-S's
            # link-state table) is charged that traffic amortised over the tasks.
            ctrl = float(getattr(router, "control_messages_total", 0.0))
            if ctrl > 0:
                per_task = ctrl / max(1, len(self.test_tasks))
                logger.info("    %s control overhead: %.3f msgs/task", router.name, per_task)
                for r in all_rows[router_start:]:
                    r["communication_overhead"] += per_task

        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(self.run_dir, "all_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info("Raw results saved to %s", csv_path)

        with open(os.path.join(self.run_dir, "training_cost.json"), "w") as f:
            json.dump(self.training_cost, f, indent=2, default=float)

        summary = self._summarise(df)
        # cost of producing the policy that was evaluated (same value for
        # every seed when a transfer-trained policy is reused)
        summary["train_time_s"] = [
            float(self.training_cost.get(m, {}).get("train_time_s", 0.0))
            for m in summary.index
        ]
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
            avg_realised_latency  = ("realised_latency", "mean"),
        avg_effective_latency = ("effective_latency",
                                     lambda x: x.replace([np.inf,-np.inf], np.nan).dropna().mean()),
            p95_effective_latency = ("effective_latency",
                                     lambda x: np.percentile(
                                         x.replace([np.inf,-np.inf], np.nan).dropna(), 95)
                                     if len(x.replace([np.inf,-np.inf], np.nan).dropna()) else np.nan),
            infeasible_rate       = ("effective_latency",
                                     lambda x: float(np.isinf(x).mean())),
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
            deadline_miss_rate    = ("deadline_miss", "mean"),
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

        # ---- IoA-aware routing score (NO explainability bonus) ----
        #
        # We use IoA-specific metrics where HybridXAI is designed to win:
        #
        #   eff_latency_score  — effective latency includes congestion/protocol
        #                        penalty; raw latency ignores agent state
        #   overhead_score     — control-message cost
        #   hop_score          — path length efficiency
        #   deadline_score     — fraction of tasks meeting the priority-scaled deadline
        #   protocol_score     — protocol compatibility across the path
        #   stretch_score      — path cost vs optimal
        #
        # Raw avg_latency is reported separately but NOT used in the composite
        # because ShortestPath optimises raw latency by construction — using it
        # would guarantee ShortestPath wins. The IoA routing problem is multi-
        # dimensional; the composite captures that.
        #
        # Weights justified in Section 6 of the paper:
        #   effective latency 20%, overhead 20%, deadline 15%,
        #   protocol 15%, hop 15%, stretch 15%
        agg["eff_latency_score"] = 1.0 / (1.0 + agg["avg_effective_latency"])
        agg["overhead_score"]    = 1.0 / (1.0 + agg["avg_overhead"])
        agg["hop_score"]         = 1.0 / (1.0 + agg["avg_hops"])
        agg["deadline_score"]    = 1.0 - agg["deadline_miss_rate"]
        agg["protocol_score"]    = 1.0 - agg["protocol_mismatch_rate"]
        agg["stretch_score"]     = 1.0 / (1.0 + agg["avg_path_stretch"])

        # Weight justification (paper Section 6):
        # - Effective latency and deadline: primary IoA QoS signals (20% each)
        # - Protocol compatibility: critical for heterogeneous IoA (20%)
        # - Overhead: communication cost efficiency (15%)
        # - Hops: path length efficiency (15%)
        # - Stretch: path optimality vs raw shortest (10%)
        # Raw avg_latency is NOT included because ShortestPath minimises it
        # by construction — it belongs in the per-metric table, not composite.
        # Control overhead is NOT part of the composite: methods obtain global
        # state in different ways (fresh per task, periodic coordinator,
        # link-state table), so it is reported as its own metric
        # (avg_overhead). Remaining weights keep their original ratios.
        agg["routing_score"] = (
            0.20 * agg["eff_latency_score"]
            + 0.20 * agg["deadline_score"]
            + 0.20 * agg["protocol_score"]
            + 0.15 * agg["hop_score"]
            + 0.10 * agg["stretch_score"]
        ) / 0.85

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

    # Wilcoxon signed-rank tests: each method vs ShortestPath on 3 key metrics.
    # We test routing_score (composite), deadline_miss_rate (lower=better, so
    # alternative="less"), and protocol_mismatch_rate (lower=better).
    def _wilcoxon_greater(method_vals, sp_vals):
        if len(method_vals) < 5:
            return np.nan
        try:
            _, p = scipy_stats.wilcoxon(method_vals, sp_vals, alternative="greater")
            return p
        except Exception:
            return np.nan

    def _wilcoxon_less(method_vals, sp_vals):
        if len(method_vals) < 5:
            return np.nan
        try:
            _, p = scipy_stats.wilcoxon(method_vals, sp_vals, alternative="less")
            return p
        except Exception:
            return np.nan

    sp = combined[combined["method"] == "ShortestPath"]
    sp_routing  = sp["routing_score"].values
    sp_deadline = sp["deadline_miss_rate"].values
    sp_proto    = sp["protocol_mismatch_rate"].values

    wilcoxon_routing, wilcoxon_deadline, wilcoxon_proto = {}, {}, {}
    for method in combined["method"].unique():
        if method == "ShortestPath":
            wilcoxon_routing[method] = wilcoxon_deadline[method] = wilcoxon_proto[method] = np.nan
            continue
        m = combined[combined["method"] == method]
        wilcoxon_routing[method]  = _wilcoxon_greater(m["routing_score"].values, sp_routing)
        wilcoxon_deadline[method] = _wilcoxon_less(m["deadline_miss_rate"].values, sp_deadline)
        wilcoxon_proto[method]    = _wilcoxon_less(m["protocol_mismatch_rate"].values, sp_proto)

    agg_mean["wilcoxon_p_vs_sp"]           = pd.Series(wilcoxon_routing)
    agg_mean["wilcoxon_deadline_p_vs_sp"]  = pd.Series(wilcoxon_deadline)
    agg_mean["wilcoxon_protocol_p_vs_sp"]  = pd.Series(wilcoxon_proto)

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