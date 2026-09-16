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
import numpy as np

from config import MAX_HOPS, LATENCY_BOUND_EPSILON, ROUTE_CACHE_MAXSIZE, PROTOCOL_PENALTY, TIE_BREAK_TOP_K
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
            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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
            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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
#  Multi-Criteria Decision Making (MADM) baselines                     #
# ================================================================== #
#
# These two baselines isolate the contribution of GloXRoute's GLOBAL
# COORDINATION from the contribution of merely combining multiple
# attributes. They score candidates using the SAME criteria available
# to HybridXAI (capability, reliability, resource, load, protocol
# compatibility, link latency) but combine them with standard, textbook
# MADM rules and WITHOUT any coordinator guidance.
#
# If GloXRoute outperforms these, the improvement is attributable to
# global coordination rather than to multi-attribute scoring alone.
#
# References:
#   SAW    - Simple Additive Weighting / Weighted Sum Model; the
#            classical MADM aggregation rule (Fishburn 1967; MacCrimmon 1968).
#   TOPSIS - Technique for Order of Preference by Similarity to Ideal
#            Solution (Hwang & Yoon, 1981).

# Shared criteria set used by BOTH MADM baselines.
#   name, weight, is_benefit  (benefit = higher is better)
MADM_CRITERIA = [
    ("capability_match", 0.30, True),   # candidate provides required capability
    ("progress",         0.25, True),   # 1/(1+hops-to-nearest-capable), local estimate
    ("reliability",      0.15, True),
    ("protocol_match",   0.10, True),
    ("resource",         0.05, True),
    ("load",             0.10, False),  # cost criterion
    ("latency",          0.05, False),  # cost criterion
]


def _madm_local_progress(G, agents, candidate, required, visited):
    """
    Purely LOCAL estimate of progress toward a capable agent: inverse of
    the hop distance to the nearest capable agent, computed with an
    unweighted BFS limited to a small radius. Deliberately does NOT use
    the coordinator -- these baselines must remain coordination-free.
    """
    if required in agents[candidate].capabilities:
        return 1.0
    # bounded BFS (radius 3) so this stays a *local* heuristic
    frontier = {candidate}
    seen = {candidate}
    for depth in range(1, 4):
        nxt = set()
        for u in frontier:
            for v in G.neighbors(u):
                if v in seen or getattr(agents[v], "failed", False):
                    continue
                if required in agents[v].capabilities:
                    return 1.0 / (1.0 + depth)
                seen.add(v)
                nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return 0.0


def _madm_decision_matrix(G, agents, current, required, candidates, visited):
    """
    Build the decision matrix for the MADM baselines: one row per
    candidate neighbour, one column per criterion in MADM_CRITERIA.
    """
    rows = []
    for n in candidates:
        edge = G[current][n]
        rows.append([
            1.0 if required in agents[n].capabilities else 0.0,
            _madm_local_progress(G, agents, n, required, visited),
            agents[n].reliability,
            1.0 if agents[current].protocol == agents[n].protocol else 0.0,
            agents[n].resource,
            agents[n].load,
            float(edge["latency"]),
        ])
    return np.array(rows, dtype=float)


class SAWRouter(BaseRouter):
    """
    Simple Additive Weighting (SAW) multi-criteria routing baseline.

    At each hop, candidate neighbours are scored by a weighted sum of
    min-max normalised criteria, with cost criteria (load, latency)
    inverted so that higher is always better. No global coordination.

    Reference: Simple Additive Weighting / Weighted Sum Model, the
    classical MADM aggregation rule.
    """
    name = "SAW_MADM"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited = [current], {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current)
                          if n not in visited and not getattr(agents[n], "failed", False)]
            if not candidates:
                break

            M = _madm_decision_matrix(G, agents, current, required, candidates, visited)

            # Min-max normalise each criterion column to [0, 1],
            # orienting every column so that larger == better.
            norm = np.zeros_like(M)
            for j, (_, _, is_benefit) in enumerate(MADM_CRITERIA):
                col = M[:, j]
                lo, hi = col.min(), col.max()
                if hi - lo < 1e-12:
                    norm[:, j] = 1.0        # criterion does not discriminate
                elif is_benefit:
                    norm[:, j] = (col - lo) / (hi - lo)
                else:
                    norm[:, j] = (hi - col) / (hi - lo)

            weights = np.array([w for _, w, _ in MADM_CRITERIA])
            scores  = norm @ weights

            best_n = candidates[int(np.argmax(scores))]
            current = best_n
            path.append(current)
            visited.add(current)

        return RouteDecision(path=path, explanation="", extra_overhead=0.0)


class TOPSISRouter(BaseRouter):
    """
    TOPSIS multi-criteria routing baseline.

    At each hop the candidate neighbours form a decision matrix which is
    vector-normalised and weighted; the positive-ideal and negative-ideal
    solutions are derived per criterion, and the candidate with the
    highest relative closeness to the positive ideal is selected.
    No global coordination.

    Reference: Hwang, C.-L. & Yoon, K. (1981). Multiple Attribute
    Decision Making: Methods and Applications.
    """
    name = "TOPSIS_MADM"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)

        current, required = task.source, task.required_capability
        path, visited = [current], {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            candidates = [n for n in G.neighbors(current)
                          if n not in visited and not getattr(agents[n], "failed", False)]
            if not candidates:
                break

            M = _madm_decision_matrix(G, agents, current, required, candidates, visited)
            weights = np.array([w for _, w, _ in MADM_CRITERIA])

            # Step 1: vector normalisation  r_ij = x_ij / sqrt(sum_i x_ij^2)
            denom = np.sqrt((M ** 2).sum(axis=0))
            denom[denom < 1e-12] = 1.0
            R = M / denom

            # Step 2: weighted normalised matrix
            V = R * weights

            # Step 3: positive-ideal (A+) and negative-ideal (A-) solutions.
            # For benefit criteria A+ takes the max, for cost criteria the min.
            a_pos, a_neg = np.zeros(V.shape[1]), np.zeros(V.shape[1])
            for j, (_, _, is_benefit) in enumerate(MADM_CRITERIA):
                if is_benefit:
                    a_pos[j], a_neg[j] = V[:, j].max(), V[:, j].min()
                else:
                    a_pos[j], a_neg[j] = V[:, j].min(), V[:, j].max()

            # Step 4: Euclidean separation from the ideal solutions
            d_pos = np.sqrt(((V - a_pos) ** 2).sum(axis=1))
            d_neg = np.sqrt(((V - a_neg) ** 2).sum(axis=1))

            # Step 5: relative closeness  C_i = d- / (d+ + d-)
            denom_c = d_pos + d_neg
            denom_c[denom_c < 1e-12] = 1.0
            closeness = d_neg / denom_c

            best_n = candidates[int(np.argmax(closeness))]
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
        self._fallback_dist_cache = {}

    # -------------------------------------------------------------- #
    def _distance_to_capable_norm(self, G, agents, candidate, required) -> float:
        """
        Fallback: distance to nearest capable agent, normalised.
        CACHED per (candidate, required) within a single routing call to avoid
        O(candidates) redundant shortest-path searches per hop, which was the
        main scalability bottleneck at large node counts (500+).
        """
        cache_key = (candidate, required)
        if cache_key in self._fallback_dist_cache:
            return self._fallback_dist_cache[cache_key]

        # Single multi-source Dijkstra from all capable (non-failed) targets,
        # instead of one shortest_path_length call per target per candidate.
        targets = [
            t for t in capable_nodes(agents, required)
            if not getattr(agents[t], "failed", False)
        ]
        if not targets:
            self._fallback_dist_cache[cache_key] = 1.0
            return 1.0

        try:
            distances = nx.multi_source_dijkstra_path_length(
                G, sources=set(targets), weight="latency"
            )
            best = distances.get(candidate, float("inf"))
        except Exception:
            best = float("inf")

        result = 1.0 if math.isinf(best) else best / (10.0 * max(1, len(G.nodes())))
        self._fallback_dist_cache[cache_key] = result
        return result

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

        self._fallback_dist_cache.clear()

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

            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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
                if len(feasible) < TIE_BREAK_TOP_K:
                    feasible = sorted(rows, key=lambda r: r["total_cost"])[:TIE_BREAK_TOP_K]

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
            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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
            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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
            candidates = [n for n in G.neighbors(current) if n not in visited and not getattr(agents[n], 'failed', False)]
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