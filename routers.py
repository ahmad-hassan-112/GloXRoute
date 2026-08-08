import networkx as nx
from typing import Dict, Optional

from config import MAX_HOPS
from models import Agent, Task, RouteDecision
from utils import capable_nodes, path_latency
from coordinator import GlobalCoordinator
from learning import FeatureExtractor, HybridParameterLearner
from nn_learning import NNFeatureExtractor, NeuralHybridLearner
from gnn_learning import GNNHybridLearner
from utils import source_can_solve, effective_edge_weight

import math
from collections import OrderedDict
from config import LATENCY_BOUND_EPSILON, ROUTE_CACHE_MAXSIZE


class BaseRouter:
    name = "BaseRouter"

    def route(self, G: nx.Graph, agents: Dict[int, Agent], task: Task) -> RouteDecision:
        raise NotImplementedError

class RouteCache:
    def __init__(self, maxsize=8192):
        self.maxsize = maxsize
        self.data = OrderedDict()
        self.hits = 0
        self.misses = 0

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


class ShortestPathRouter(BaseRouter):
    name = "ShortestPath"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        source = task.source
        required = task.required_capability
        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_cost = float("inf")

        for target in candidates:
            try:
                weight = effective_edge_weight(agents)
                path = nx.shortest_path(G, source=task.source, target=target, weight=weight)
                cost = path_latency(G, agents, path)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            except nx.NetworkXNoPath:
                continue

        return RouteDecision(path=best_path)


class CentralizedRouter(BaseRouter):
    name = "Centralized"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        source = task.source
        required = task.required_capability
        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_score = float("inf")

        for target in candidates:
            try:
                weight = effective_edge_weight(agents)
                path = nx.shortest_path(G, source=task.source, target=target, weight=weight)

                latency = path_latency(G, agents, path)
                load_penalty = agents[target].load * 5.0
                reliability_bonus = agents[target].reliability * 3.0
                resource_bonus = agents[target].resource * 2.0

                score = latency + load_penalty - reliability_bonus - resource_bonus

                if score < best_score:
                    best_score = score
                    best_path = path

            except nx.NetworkXNoPath:
                continue

        return RouteDecision(path=best_path, extra_overhead=5.0)


class DecentralizedGreedyRouter(BaseRouter):
    name = "DecentralizedGreedy"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]

            if not candidates:
                break

            best_neighbor = None
            best_score = -float("inf")

            for n in candidates:
                capability_match = 1.0 if required in agents[n].capabilities else 0.0
                reliability = agents[n].reliability
                load = agents[n].load
                latency = G[current][n]["latency"]

                score = (
                    4.0 * capability_match
                    + 2.0 * reliability
                    - 2.0 * load
                    - 0.15 * latency
                )

                if score > best_score:
                    best_score = score
                    best_neighbor = n

            current = best_neighbor
            path.append(current)
            visited.add(current)

        return RouteDecision(path=path)


class BiRouterStyleRouter(BaseRouter):
    name = "BiRouterStyle"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}

        alpha = 0.6

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]

            if not candidates:
                break

            best_neighbor = None
            best_score = -float("inf")

            for n in candidates:
                # Importance score
                imp_score = 1.0 if required in agents[n].capabilities else 0.0

                # Limited two-hop awareness
                two_hop_neighbors = list(G.neighbors(n))
                two_hop_match = any(
                    required in agents[x].capabilities
                    for x in two_hop_neighbors
                )

                if two_hop_match:
                    imp_score += 0.5

                # Gap score
                latency = G[current][n]["latency"]
                gap_score = agents[n].reliability - agents[n].load - 0.1 * latency

                # Reputation
                reputation = agents[n].reliability

                final_score = reputation * (
                    alpha * imp_score + (1.0 - alpha) * gap_score
                )

                if final_score > best_score:
                    best_score = final_score
                    best_neighbor = n

            current = best_neighbor
            path.append(current)
            visited.add(current)

        return RouteDecision(path=path)


class HybridXAIRouter(BaseRouter):
    name = "HybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, router_name: str = "HybridXAI"):
        self.coordinator = coordinator
        self.route_cache = RouteCache(maxsize=ROUTE_CACHE_MAXSIZE)
        self.name = router_name
        self.min_confidence_score = 0.5

    def _distance_to_capable_norm(self, G, agents, candidate, required):
        targets = capable_nodes(agents, required)
        best_distance = float("inf")

        for target in targets:
            try:
                distance = nx.shortest_path_length(
                    G,
                    candidate,
                    target,
                    weight="latency",
                )
                if distance < best_distance:
                    best_distance = distance
            except nx.NetworkXNoPath:
                continue

        if best_distance == float("inf"):
            return 1.0

        return best_distance / (10.0 * max(1, len(G.nodes())))

    def _fallback_next_hop(self, G, agents, current, required, visited):
        recommended = self.coordinator.next_hop.get(current, {}).get(required)
        if recommended is not None and recommended not in visited:
            return recommended, "fallback=coordinator_recommended_next_hop"

        targets = capable_nodes(agents, required)
        best_path = None
        best_cost = float("inf")

        for target in targets:
            try:
                path = nx.shortest_path(G, current, target, weight="latency")
                if len(path) <= 1:
                    continue

                cost = nx.shortest_path_length(G, current, target, weight="latency")
                next_hop = path[1]

                if next_hop in visited:
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            except nx.NetworkXNoPath:
                continue

        if best_path is not None:
            return best_path[1], "fallback=shortest_path_to_capable"

        return None, "fallback=none"

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )

        cache_key = (
            int(task.source),
            str(task.required_capability),
            int(getattr(self.coordinator, "state_version", 0)),
        )

        cached_path = self.route_cache.get(cache_key)

        if cached_path is not None:
            return RouteDecision(
                path=cached_path,
                explanation="Cached GloXRoute-LB route reused from current coordinator state.",
                extra_overhead=0.1,
            )

        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}
        explanations = []

        weight_fn = effective_edge_weight(agents)

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]

            if not candidates:
                break

            recommended_next = self.coordinator.next_hop.get(current, {}).get(required)
            coordinator_context = self.coordinator.explain_guidance(current, required)

            candidate_rows = []

            for n in candidates:
                edge_data = G[current][n]

                step_cost = weight_fn(current, n, edge_data)

                if step_cost is None:
                    continue

                distance_table = getattr(self.coordinator, "distance", {})
                remaining_cost = distance_table.get(n, {}).get(
                    required,
                    float("inf")
                )

                # Fallback if coordinator.distance is missing or incomplete.
                if math.isinf(remaining_cost):
                    distance_to_required_norm = self._distance_to_capable_norm(
                        G,
                        agents,
                        n,
                        required,
                    )
                    remaining_cost = distance_to_required_norm * 10.0
                else:
                    distance_to_required_norm = min(remaining_cost / 20.0, 1.0)

                if math.isinf(remaining_cost):
                    continue

                total_cost = step_cost + remaining_cost

                capability_match = 1.0 if required in agents[n].capabilities else 0.0
                recommended_match = 1.0 if n == recommended_next else 0.0
                global_hint = self.coordinator.guidance.get(n, {}).get(required, 0.0)

                reliability = agents[n].reliability
                resource = agents[n].resource
                load = agents[n].load
                latency = G[current][n]["latency"]
                congestion = self.coordinator.congestion_penalty(n)

                onward_unvisited = sum(
                    1 for nn in G.neighbors(n)
                    if nn not in visited and nn != current
                )
                dead_end_penalty = (
                    1.0 if onward_unvisited == 0 and capability_match == 0.0 else 0.0
                )

                candidate_rows.append({
                    "node": n,
                    "step_cost": step_cost,
                    "remaining_cost": remaining_cost,
                    "total_cost": total_cost,
                    "capability_match": capability_match,
                    "recommended_match": recommended_match,
                    "global_hint": global_hint,
                    "reliability": reliability,
                    "resource": resource,
                    "load": load,
                    "latency": latency,
                    "congestion": congestion,
                    "distance_to_required_norm": distance_to_required_norm,
                    "dead_end_penalty": dead_end_penalty,
                })

            if not candidate_rows:
                fallback_neighbor, fallback_reason = self._fallback_next_hop(
                    G,
                    agents,
                    current,
                    required,
                    visited,
                )

                if fallback_neighbor is None:
                    break

                best_neighbor = fallback_neighbor
                best_explanation = (
                    f"from={current}->to={best_neighbor}; {fallback_reason}; "
                    f"coordinator_context=({coordinator_context})"
                )

            else:
                # Step 1: find the lowest estimated latency route.
                best_total_cost = min(row["total_cost"] for row in candidate_rows)

                # Step 2: keep only candidates close to the best latency route.
                latency_bound = best_total_cost * (1.0 + LATENCY_BOUND_EPSILON)

                feasible_rows = [
                    row for row in candidate_rows
                    if row["total_cost"] <= latency_bound
                ]

                # Step 3: among near-best latency choices, select by quality/XAI factors.
                for row in feasible_rows:
                    row["tie_score"] = (
                        3.0 * row["capability_match"]
                        + 2.0 * row["recommended_match"]
                        + 2.0 * row["global_hint"]
                        + 0.75 * row["reliability"]
                        + 0.25 * row["resource"]
                        - 0.50 * row["load"]
                        - 0.25 * row["congestion"]
                        - 0.50 * row["dead_end_penalty"]
                    )

                winner = max(feasible_rows, key=lambda row: row["tie_score"])

                best_neighbor = winner["node"]

                best_explanation = (
                    f"from={current}->to={best_neighbor}; "
                    f"coordinator_context=({coordinator_context}); "
                    f"selection=latency_bounded_xai; "
                    f"step_cost={winner['step_cost']:.3f}; "
                    f"remaining_cost={winner['remaining_cost']:.3f}; "
                    f"total_cost={winner['total_cost']:.3f}; "
                    f"latency_bound={latency_bound:.3f}; "
                    f"capability_match={winner['capability_match']}; "
                    f"recommended_by_global={winner['recommended_match']}; "
                    f"global_guidance={winner['global_hint']:.3f}; "
                    f"reliability={winner['reliability']:.3f}; "
                    f"resource={winner['resource']:.3f}; "
                    f"load={winner['load']:.3f}; "
                    f"latency={winner['latency']:.3f}; "
                    f"congestion={winner['congestion']:.3f}; "
                    f"dist_to_required_norm={winner['distance_to_required_norm']:.3f}; "
                    f"dead_end_penalty={winner['dead_end_penalty']:.3f}; "
                    f"tie_score={winner['tie_score']:.3f}"
                )

            current = best_neighbor
            path.append(current)
            visited.add(current)
            explanations.append(best_explanation)

            if required in agents[current].capabilities:
                break

        # Cache only successful routes.
        if path and required in agents[path[-1]].capabilities:
            self.route_cache.put(cache_key, path)

        return RouteDecision(
            path=path,
            explanation=" | ".join(explanations),
            extra_overhead=0.1,
        )

class LearnedHybridXAIRouter(BaseRouter):
    name = "LearnedHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner: HybridParameterLearner):
        self.coordinator = coordinator
        self.learner = learner
        self.extractor = FeatureExtractor()

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}
        explanations = []

        if required in agents[current].capabilities:
            explanation = (
                f"source_agent={current} already has required capability={required}; "
                f"no routing needed"
            )

            return RouteDecision(
                path=path,
                explanation=explanation,
                extra_overhead=0.1
            )

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]

            if not candidates:
                break

            best_neighbor = None
            best_score = -float("inf")
            best_explanation = ""

            for candidate in candidates:
                features = self.extractor.extract(
                    G,
                    agents,
                    self.coordinator,
                    current,
                    candidate,
                    required,
                )

                score = self.learner.predict_score(features)

                xai_reason = self.learner.explain(features)

                explanation = (
                    f"from={current}->to={candidate}; "
                    f"learned_score={score:.3f}; "
                    f"{xai_reason}"
                )

                if score > best_score:
                    best_score = score
                    best_neighbor = candidate
                    best_explanation = explanation

            current = best_neighbor
            path.append(current)
            visited.add(current)
            explanations.append(best_explanation)

            if required in agents[current].capabilities:
                break

        return RouteDecision(
            path=path,
            explanation=" | ".join(explanations),
            extra_overhead=0.1
        )
    

class NeuralHybridXAIRouter(BaseRouter):
    name = "NeuralHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner: NeuralHybridLearner):
        self.coordinator = coordinator
        self.learner = learner
        self.extractor = NNFeatureExtractor()

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}
        explanations = []

        if required in agents[current].capabilities:
            explanation = (
                f"source_agent={current} already has required capability={required}; "
                f"no routing needed"
            )

            return RouteDecision(
                path=path,
                explanation=explanation,
                extra_overhead=0.1
            )

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]

            if not candidates:
                break

            best_neighbor = None
            best_score = -float("inf")
            best_explanation = ""

            for candidate in candidates:
                features = self.extractor.extract(
                    G=G,
                    agents=agents,
                    coordinator=self.coordinator,
                    current_node=current,
                    candidate_node=candidate,
                    required_capability=required,
                )

                score = self.learner.predict_score(features)
                xai_reason = self.learner.explain(features)

                explanation = (
                    f"from={current}->to={candidate}; "
                    f"nn_score={score:.3f}; "
                    f"{xai_reason}"
                )

                if score > best_score:
                    best_score = score
                    best_neighbor = candidate
                    best_explanation = explanation

            current = best_neighbor
            path.append(current)
            visited.add(current)
            explanations.append(best_explanation)

            if required in agents[current].capabilities:
                break

        return RouteDecision(
            path=path,
            explanation=" | ".join(explanations),
            extra_overhead=0.1
        )


class GNNHybridXAIRouter(BaseRouter):
    name = "GNNHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner: GNNHybridLearner):
        self.coordinator = coordinator
        self.learner = learner

    def route(self, G, agents, task):
        if source_can_solve(agents, task):
            return RouteDecision(
                path=[task.source],
                explanation="",
                extra_overhead=0.0,
            )
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}
        explanations = []

        if required in agents[current].capabilities:
            explanation = (
                f"source_agent={current} already has required capability={required}; "
                f"no routing needed"
            )
            return RouteDecision(path=path, explanation=explanation, extra_overhead=0.1)

        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break

            neighbors = list(G.neighbors(current))
            candidates = [n for n in neighbors if n not in visited]
            if not candidates:
                break

            best_neighbor = None
            best_score = -float("inf")
            best_explanation = ""

            for candidate in candidates:
                score = self.learner.predict_score(current, candidate, required)
                xai_reason = self.learner.explain(current, candidate, required)
                explanation = (
                    f"from={current}->to={candidate}; "
                    f"gnn_score={score:.3f}; "
                    f"{xai_reason}"
                )

                if score > best_score:
                    best_score = score
                    best_neighbor = candidate
                    best_explanation = explanation

            if best_neighbor is None:
                break

            current = best_neighbor
            path.append(current)
            visited.add(current)
            explanations.append(best_explanation)

            if required in agents[current].capabilities:
                break

        return RouteDecision(
            path=path,
            explanation=" | ".join(explanations),
            extra_overhead=0.1
        )