import networkx as nx
from typing import Dict, Optional

from config import MAX_HOPS
from models import Agent, Task, RouteDecision
from utils import capable_nodes, path_latency
from coordinator import GlobalCoordinator
from learning import FeatureExtractor, HybridParameterLearner
from nn_learning import NNFeatureExtractor, NeuralHybridLearner
from gnn_learning import GNNHybridLearner


class BaseRouter:
    name = "BaseRouter"

    def route(self, G: nx.Graph, agents: Dict[int, Agent], task: Task) -> RouteDecision:
        raise NotImplementedError


class ShortestPathRouter(BaseRouter):
    name = "ShortestPath"

    def route(self, G, agents, task):
        source = task.source
        required = task.required_capability
        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_cost = float("inf")

        for target in candidates:
            try:
                path = nx.shortest_path(G, source, target, weight="latency")
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
        source = task.source
        required = task.required_capability
        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_score = float("inf")

        for target in candidates:
            try:
                path = nx.shortest_path(G, source, target, weight="latency")

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
        current = task.source
        required = task.required_capability

        path = [current]
        visited = {current}
        explanations = []

        # If source already has required capability, explain it.
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

            recommended_next = self.coordinator.next_hop.get(current, {}).get(required)
            coordinator_context = self.coordinator.explain_guidance(current, required)

            best_neighbor = None
            best_score = -float("inf")
            best_explanation = ""

            for n in candidates:
                capability_match = 1.0 if required in agents[n].capabilities else 0.0
                recommended_match = 1.0 if n == recommended_next else 0.0

                global_hint = self.coordinator.guidance.get(n, {}).get(required, 0.0)
                reliability = agents[n].reliability
                resource = agents[n].resource
                load = agents[n].load
                latency = G[current][n]["latency"]

                # Normalize latency because latency is between 1 and 10.
                latency_norm = latency / 10.0

                # Do not double-count load and congestion too strongly.
                congestion = self.coordinator.congestion_penalty(n)
                distance_to_required_norm = self._distance_to_capable_norm(
                    G,
                    agents,
                    n,
                    required,
                )

                onward_unvisited = sum(
                    1 for nn in G.neighbors(n)
                    if nn not in visited and nn != current
                )
                dead_end_penalty = 1.0 if onward_unvisited == 0 and capability_match == 0.0 else 0.0

                score = (
                    5.0 * capability_match
                    + 3.0 * recommended_match
                    + 2.0 * global_hint
                    + 1.5 * reliability
                    + 1.0 * resource
                    - 2.0 * load
                    - 1.0 * congestion
                    - 3.0 * latency_norm
                    - 4.0 * distance_to_required_norm
                    - 2.0 * dead_end_penalty
                )

                explanation = (
                    f"from={current}->to={n}; "
                    f"coordinator_context=({coordinator_context}); "
                    f"capability_match={capability_match}; "
                    f"recommended_by_global={recommended_match}; "
                    f"global_guidance={global_hint:.3f}; "
                    f"reliability={reliability:.3f}; "
                    f"resource={resource:.3f}; "
                    f"load={load:.3f}; "
                    f"latency={latency:.3f}; "
                    f"dist_to_required_norm={distance_to_required_norm:.3f}; "
                    f"dead_end_penalty={dead_end_penalty:.3f}; "
                    f"score={score:.3f}"
                )

                if score > best_score:
                    best_score = score
                    best_neighbor = n
                    best_explanation = explanation

            used_fallback = False
            if best_neighbor is None or best_score < self.min_confidence_score:
                fallback_neighbor, fallback_reason = self._fallback_next_hop(
                    G,
                    agents,
                    current,
                    required,
                    visited,
                )
                if fallback_neighbor is not None:
                    best_neighbor = fallback_neighbor
                    used_fallback = True
                    best_explanation = (
                        f"from={current}->to={best_neighbor}; {fallback_reason}; "
                        f"trigger_score={best_score:.3f}; "
                        f"coordinator_context=({coordinator_context})"
                    )

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
    

class LearnedHybridXAIRouter(BaseRouter):
    name = "LearnedHybridXAI"

    def __init__(self, coordinator: GlobalCoordinator, learner: HybridParameterLearner):
        self.coordinator = coordinator
        self.learner = learner
        self.extractor = FeatureExtractor()

    def route(self, G, agents, task):
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