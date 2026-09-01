"""
routers.py — GloXRoute routing algorithms.

Main comparison (5 methods):
  ShortestPathRouter         — baseline: Dijkstra to nearest capable agent
  CentralizedRouter          — baseline: global optimiser (high overhead)
  DecentralizedGreedyRouter  — baseline: local greedy, no coordination
  BiRouterStyleRouter        — literature-inspired baseline (local bi-criteria)
  HybridXAIRouter            — PROPOSED: hybrid local+global, glass-box

Ablation-only (not in main table, run separately):
  LearnedHybridXAIRouter     — ablation: learned scoring weights
  NeuralHybridXAIRouter      — ablation: neural scoring (requires torch)
  GNNHybridXAIRouter         — ablation: GNN scoring (requires torch)

Protocol term: added to HybridXAIRouter scoring so protocol_mismatch_rate
can be reduced by the proposed method (was 58-69% before because no router
considered protocols).
"""

import math
from collections import OrderedDict
from typing import Dict, Optional

import networkx as nx

from config import MAX_HOPS, LATENCY_BOUND_EPSILON, ROUTE_CACHE_MAXSIZE, PROTOCOL_PENALTY
from coordinator import GlobalCoordinator
from models import Agent, Task, RouteDecision
from utils import capable_nodes, path_latency, source_can_solve, effective_edge_weight


# ================================================================== #
#  Utilities                                                           #
# ================================================================== #

class RouteCache:
    """LRU cache keyed on (source, capability, coordinator_version)."""

    def __init__(self, maxsize: int = ROUTE_CACHE_MAXSIZE):
        self.maxsize = maxsize
        self.data    = OrderedDict()
        self.hits    = 0
        self.misses  = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return list(self.data[key])

    def put(self, key, path):
        self.data[key] = list(path)
        self.data.move_to_end(key)
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)


class BaseRouter:
    name = "BaseRouter"

    def route(self, G: nx.Graph, agents: Dict[int, Agent], task: Task) -> RouteDecision:
        raise NotImplementedError


# ================================================================== #
#  Baselines                                                           #
# ================================================================== #

class ShortestPathRouter(BaseRouter):
    """
    Dijkstra to nearest capable agent (by effective edge weight).
    Reference: Dijkstra (1959). Numerische Mathematik, 1(1), 269-271.
    """
    name = "ShortestPath"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        weight_fn = effective_edge_weight(agents)
        best_path, best_cost = [task.source], float("inf")

        for target in capable_nodes(agents, task.required_capability):
            try:
                path = nx.shortest_path(G, task.source, target, weight=weight_fn)
                cost = path_latency(G, agents, path)
                if cost < best_cost:
                    best_cost, best_path = cost, path
            except nx.NetworkXNoPath:
                continue

        return RouteDecision(path=best_path, explanation="", extra_overhead=0.0)


class CentralizedRouter(BaseRouter):
    """
    Global optimiser: considers load, reliability, resource at destination.
    High communication overhead (requires full global state).
    """
    name = "Centralized"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        weight_fn = effective_edge_weight(agents)
        best_path, best_score = [task.source], float("inf")

        for target in capable_nodes(agents, task.required_capability):
            try:
                path = nx.shortest_path(G, task.source, target, weight=weight_fn)
                latency         = path_latency(G, agents, path)
                load_penalty    = agents[target].load       * 5.0
                rel_bonus       = agents[target].reliability * 3.0
                res_bonus       = agents[target].resource    * 2.0
                score           = latency + load_penalty - rel_bonus - res_bonus
                if score < best_score:
                    best_score, best_path = score, path
            except nx.NetworkXNoPath:
                continue

        # Fixed overhead: Centralized must query the full network
        return RouteDecision(path=best_path, explanation="", extra_overhead=float(len(G.nodes())))


class DecentralizedGreedyRouter(BaseRouter):
    """
    Local greedy: at each hop selects the neighbour with the best local score.
    No global information. Standard literature baseline.
    Reference: Karp & Kung (2000). GPSR, MobiCom.
    """
    name = "DecentralizedGreedy"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited = [current], {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            best_n, best_s = None, -float("inf")
            for n in candidates:
                cap_match = 1.0 if required in agents[n].capabilities else 0.0
                score = (
                    4.0 * cap_match
                    + 2.0 * agents[n].reliability
                    - 2.0 * agents[n].load
                    - 0.15 * G[current][n]["latency"]
                )
                if score > best_s:
                    best_s, best_n = score, n

            current = best_n
            path.append(current)
            visited.add(current)

        return RouteDecision(path=path, explanation="", extra_overhead=0.0)


class BiRouterStyleRouter(BaseRouter):
    """
    Literature-inspired bi-criteria baseline adapted to this simulator.
    Balances a long-term Importance score (capability relevance) and a
    short-term Gap score (contextual continuity) analogous to BiRouter
    (Yang et al., AAAI-26, arXiv:2512.00740).

    NOTE: This is NOT a reproduction of the original BiRouter (which
    operates on LLM agent chains with semantic embeddings). It is a
    structural adaptation of its dual-score philosophy to the abstract
    graph routing setting used in this paper, implemented without LLMs
    or training data.
    """
    name = "BiRouterStyle"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited = [current], {current}
        alpha = 0.6   # balance importance vs gap

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            best_n, best_s = None, -float("inf")
            for n in candidates:
                # Importance: capability relevance (long-term heuristic)
                imp = 1.0 if required in agents[n].capabilities else 0.0
                if any(required in agents[x].capabilities for x in G.neighbors(n)):
                    imp += 0.5

                # Gap: contextual fit (short-term)
                gap = agents[n].reliability - agents[n].load - 0.1 * G[current][n]["latency"]

                # Reputation gate
                rep   = agents[n].reliability
                score = rep * (alpha * imp + (1.0 - alpha) * gap)

                if score > best_s:
                    best_s, best_n = score, n

            current = best_n
            path.append(current)
            visited.add(current)

        return RouteDecision(path=path, explanation="", extra_overhead=0.0)


# ================================================================== #
#  Proposed: HybridXAI                                                #
# ================================================================== #

class HybridXAIRouter(BaseRouter):
    """
    GloXRoute — Proposed Method.

    At each hop:
      1. Obtain global coordinator guidance (lazy, cached).
      2. Filter candidates to those within LATENCY_BOUND_EPSILON of
         the lowest-cost option (so routing stays near-optimal on latency).
      3. Among near-optimal candidates, score by multi-attribute glass-box
         function including capability, reliability, load, resource,
         protocol compatibility, and global hint.
      4. Emit a human-readable explanation for every decision.

    Overhead: proportional to routing decisions made (lazy coordinator),
    not to full (node × capability) cross-product.
    """
    name = "HybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, router_name: str = "HybridXAI_HeuristicCoord"):
        self.coordinator  = coordinator
        self.route_cache  = RouteCache(maxsize=ROUTE_CACHE_MAXSIZE)
        self.name         = router_name

    # -------------------------------------------------------------- #
    def _distance_to_capable_norm(self, G, agents, candidate, required) -> float:
        """Fallback: BFS distance to nearest capable agent, normalised."""
        best = float("inf")
        for target in capable_nodes(agents, required):
            try:
                d = nx.shortest_path_length(G, candidate, target, weight="latency")
                if d < best:
                    best = d
            except nx.NetworkXNoPath:
                continue
        if math.isinf(best):
            return 1.0
        return best / (10.0 * max(1, len(G.nodes())))

    def _fallback_hop(self, G, agents, current, required, visited):
        """If no scored candidate passes filters, fall back to Dijkstra."""
        rec = self.coordinator.next_hop[current][required]
        if rec is not None and rec not in visited:
            return rec, "fallback=coordinator_next_hop"

        best_path, best_cost = None, float("inf")
        for target in capable_nodes(agents, required):
            try:
                path = nx.shortest_path(G, current, target, weight="latency")
                if len(path) <= 1:
                    continue
                cost = nx.shortest_path_length(G, current, target, weight="latency")
                nh   = path[1]
                if nh in visited:
                    continue
                if cost < best_cost:
                    best_cost, best_path = cost, path
            except nx.NetworkXNoPath:
                continue

        if best_path:
            return best_path[1], "fallback=dijkstra"
        return None, "fallback=none"

    # -------------------------------------------------------------- #
    def route(self, G, agents, task) -> RouteDecision:
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        cache_key = (
            int(task.source),
            str(task.required_capability),
            int(getattr(self.coordinator, "state_version", 0)),
        )
        cached = self.route_cache.get(cache_key)
        if cached is not None:
            return RouteDecision(
                path=cached,
                explanation="Cached GloXRoute route reused.",
                extra_overhead=0.0,
            )

        current, required  = task.source, task.required_capability
        path, visited      = [current], {current}
        explanations       = []
        weight_fn          = effective_edge_weight(agents)

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            rec_next = self.coordinator.next_hop[current][required]
            coord_ctx = self.coordinator.explain_guidance(current, required)

            rows = []
            for n in candidates:
                edge      = G[current][n]
                step_cost = weight_fn(current, n, edge)
                if step_cost is None:
                    continue

                dist_val = self.coordinator.distance[n][required]
                if math.isinf(dist_val):
                    dist_norm  = self._distance_to_capable_norm(G, agents, n, required)
                    remaining  = dist_norm * 10.0
                else:
                    dist_norm = min(dist_val / 20.0, 1.0)
                    remaining = dist_val

                if math.isinf(remaining):
                    continue

                cap_match   = 1.0 if required in agents[n].capabilities else 0.0
                rec_match   = 1.0 if n == rec_next else 0.0
                hint        = self.coordinator.guidance[n][required]
                reliability = agents[n].reliability
                resource    = agents[n].resource
                load        = agents[n].load
                latency     = edge["latency"]
                congestion  = agents[n].load   # congestion proxy

                # ---- Protocol compatibility term (NEW) ----
                # Rewards protocol-compatible neighbours to reduce mismatch rate
                proto_match = 1.0 if agents[current].protocol == agents[n].protocol else 0.0

                onward = sum(
                    1 for nn in G.neighbors(n)
                    if nn not in visited and nn != current
                )
                dead_end = 1.0 if onward == 0 and cap_match == 0.0 else 0.0

                rows.append({
                    "node": n,
                    "total_cost":         step_cost + remaining,
                    "cap_match":          cap_match,
                    "rec_match":          rec_match,
                    "hint":               hint,
                    "reliability":        reliability,
                    "resource":           resource,
                    "load":               load,
                    "latency":            latency,
                    "congestion":         congestion,
                    "proto_match":        proto_match,
                    "dist_norm":          dist_norm,
                    "dead_end":           dead_end,
                    "step_cost":          step_cost,
                    "remaining":          remaining,
                })

            if not rows:
                fb, reason = self._fallback_hop(G, agents, current, required, visited)
                if fb is None:
                    break
                best_n = fb
                expl   = (
                    f"from={current}->to={best_n}; {reason}; "
                    f"coordinator_context=({coord_ctx})"
                )
            else:
                best_total = min(r["total_cost"] for r in rows)
                bound      = best_total * (1.0 + LATENCY_BOUND_EPSILON)
                feasible   = [r for r in rows if r["total_cost"] <= bound]

                for r in feasible:
                    r["tie_score"] = (
                        3.0  * r["cap_match"]
                        + 2.0  * r["rec_match"]
                        + 2.0  * r["hint"]
                        + 1.5  * r["reliability"]
                        + 0.5  * r["resource"]
                        - 1.0  * r["load"]
                        - 0.5  * r["congestion"]
                        + 1.0  * r["proto_match"]      # protocol compatibility bonus
                        - 0.5  * r["dead_end"]
                    )

                w      = max(feasible, key=lambda r: r["tie_score"])
                best_n = w["node"]

                expl = (
                    f"from={current}->to={best_n}; "
                    f"coordinator_context=({coord_ctx}); "
                    f"cap_match={w['cap_match']}; "
                    f"recommended_by_global={w['rec_match']}; "
                    f"global_guidance={w['hint']:.3f}; "
                    f"reliability={w['reliability']:.3f}; "
                    f"resource={w['resource']:.3f}; "
                    f"load={w['load']:.3f}; "
                    f"latency={w['latency']:.3f}; "
                    f"proto_match={w['proto_match']}; "
                    f"dist_to_required_norm={w['dist_norm']:.3f}; "
                    f"dead_end_penalty={w['dead_end']}; "
                    f"tie_score={w['tie_score']:.3f}"
                )

            current = best_n
            path.append(current)
            visited.add(current)
            explanations.append(expl)

            if required in agents[current].capabilities:
                break

        if path and required in agents[path[-1]].capabilities:
            self.route_cache.put(cache_key, path)

        return RouteDecision(
            path=path,
            explanation=" | ".join(explanations),
            extra_overhead=0.1,
        )


# ================================================================== #
#  Ablation variants (not in main comparison table)                    #
# ================================================================== #

class LearnedHybridXAIRouter(BaseRouter):
    """Ablation: learned scoring weights via logistic regression."""
    name = "LearnedHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner):
        from learning import FeatureExtractor
        self.coordinator = coordinator
        self.learner     = learner
        self.extractor   = FeatureExtractor()

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited, explanations = [current], {current}, []

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            best_n, best_s, best_expl = None, -float("inf"), ""
            for n in candidates:
                feats  = self.extractor.extract(G, agents, self.coordinator, current, n, required)
                score  = self.learner.predict_score(feats)
                reason = self.learner.explain(feats)
                if score > best_s:
                    best_s, best_n = score, n
                    best_expl = f"from={current}->to={n}; learned_score={score:.3f}; {reason}"

            current = best_n
            path.append(current)
            visited.add(current)
            explanations.append(best_expl)

        return RouteDecision(path=path, explanation=" | ".join(explanations), extra_overhead=0.1)


class NeuralHybridXAIRouter(BaseRouter):
    """Ablation: neural network scoring. Requires torch."""
    name = "NeuralHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner):
        try:
            from nn_learning import NNFeatureExtractor
            self.extractor = NNFeatureExtractor()
        except ImportError:
            self.extractor = None
        self.coordinator = coordinator
        self.learner     = learner

    def route(self, G, agents, task):
        if self.extractor is None:
            return RouteDecision(path=[task.source], explanation="torch not available")
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited, explanations = [current], {current}, []

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            best_n, best_s, best_expl = None, -float("inf"), ""
            for n in candidates:
                feats  = self.extractor.extract(G=G, agents=agents, coordinator=self.coordinator,
                                                current_node=current, candidate_node=n,
                                                required_capability=required)
                score  = self.learner.predict_score(feats)
                reason = self.learner.explain(feats)
                if score > best_s:
                    best_s, best_n = score, n
                    best_expl = f"from={current}->to={n}; nn_score={score:.3f}; {reason}"

            current = best_n
            path.append(current)
            visited.add(current)
            explanations.append(best_expl)

        return RouteDecision(path=path, explanation=" | ".join(explanations), extra_overhead=0.1)


class GNNHybridXAIRouter(BaseRouter):
    """Ablation: GNN scoring. Requires torch."""
    name = "GNNHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner):
        self.coordinator = coordinator
        self.learner     = learner

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited, explanations = [current], {current}, []

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current) if n not in visited]
            if not candidates:
                break

            best_n, best_s, best_expl = None, -float("inf"), ""
            for n in candidates:
                try:
                    score  = self.learner.predict_score(current, n, required)
                    reason = self.learner.explain(current, n, required)
                except Exception:
                    score, reason = 0.0, "gnn_error"
                if score > best_s:
                    best_s, best_n = score, n
                    best_expl = f"from={current}->to={n}; gnn_score={score:.3f}; {reason}"

            if best_n is None:
                break
            current = best_n
            path.append(current)
            visited.add(current)
            explanations.append(best_expl)

        return RouteDecision(path=path, explanation=" | ".join(explanations), extra_overhead=0.1)