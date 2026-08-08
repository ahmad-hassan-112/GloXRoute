import networkx as nx
from typing import Dict
import numpy as np
import os

from config import CAPABILITIES
from models import Agent

from utils import capable_nodes, effective_edge_weight


class GlobalCoordinator:
    """
    Lightweight centralized global agent.

    It provides:
    1. global guidance score
    2. recommended next-hop direction
    """

    def __init__(self, G: nx.Graph, agents: Dict[int, Agent]):
        self.G = G
        self.agents = agents
        self.guidance = {}
        self.next_hop = {}
        self.distance = {}
        self.state_version = 0
        self.update_round = 0
        self.last_summary = {}
        self.compute_guidance()

    def compute_guidance(self):
        self.guidance = {node: {} for node in self.G.nodes()}
        self.next_hop = {node: {} for node in self.G.nodes()}
        self.distance = {node: {} for node in self.G.nodes()}

        weight = effective_edge_weight(self.agents)

        for capability in CAPABILITIES:
            targets = [
                node for node in capable_nodes(self.agents, capability)
                if not getattr(self.agents[node], "failed", False)
            ]

            if not targets:
                for node in self.G.nodes():
                    self.distance[node][capability] = float("inf")
                    self.guidance[node][capability] = 0.0
                    self.next_hop[node][capability] = None
                continue

            try:
                distances, paths = nx.multi_source_dijkstra(
                    self.G,
                    sources=set(targets),
                    weight=weight,
                )
            except Exception:
                distances, paths = {}, {}

            for node in self.G.nodes():
                if getattr(self.agents[node], "failed", False):
                    self.guidance[node][capability] = 0.0
                    self.next_hop[node][capability] = None
                    continue

                if node not in distances:
                    self.distance[node][capability] = float("inf")
                    self.guidance[node][capability] = 0.0
                    self.next_hop[node][capability] = None
                    continue

                cost = float(distances[node])
                self.distance[node][capability] = cost
                self.guidance[node][capability] = 1.0 / (1.0 + cost)

                path_from_target_to_node = paths[node]
                path_from_node_to_target = list(reversed(path_from_target_to_node))

                if len(path_from_node_to_target) > 1:
                    self.next_hop[node][capability] = path_from_node_to_target[1]
                else:
                    self.next_hop[node][capability] = node

        self.state_version += 1

    def update_guidance(self):
        # Lightweight periodic refresh from summarized current agent state.
        self.update_round += 1
        self.last_summary = self.network_summary()
        self.compute_guidance()

    def network_summary(self):
        loads = [agent.load for agent in self.agents.values()]
        resources = [agent.resource for agent in self.agents.values()]
        reliability = [agent.reliability for agent in self.agents.values()]

        return {
            "avg_load": sum(loads) / max(1, len(loads)),
            "avg_resource": sum(resources) / max(1, len(resources)),
            "avg_reliability": sum(reliability) / max(1, len(reliability)),
        }

    def congestion_penalty(self, node: int) -> float:
        return self.agents[node].load

    def explain_guidance(self, node: int, capability: str) -> str:
        agent = self.agents[node]

        return (
            f"update_round={self.update_round}, "
            f"global_hint={self.guidance[node][capability]:.3f}, "
            f"recommended_next_hop={self.next_hop[node][capability]}, "
            f"load={agent.load:.3f}, "
            f"reliability={agent.reliability:.3f}, "
            f"resource={agent.resource:.3f}"
        )

class LearnedGlobalCoordinator(GlobalCoordinator):
    """
    Enhanced Trainable Coordinator:
    - keeps heuristic next-hop planner from GlobalCoordinator
    - learns a lightweight regression model for guidance scores
    - improved features and dynamic blending
    """

    def __init__(self, G: nx.Graph, agents: Dict[int, Agent], blend_alpha: float = 0.7):
        self.blend_alpha = blend_alpha
        self.capability_weights = {}
        self.is_trained = False
        self.dynamic_alpha = blend_alpha
        self.feature_names = [
            "bias",
            "reliability",
            "resource",
            "load",
            "degree_norm",
            "has_capability",
            "capable_neighbor_ratio",
            "neighbor_load",
            "avg_neighbor_reliability",
            "num_neighbors",
        ]
        self.feature_schema_version = "coord_v1"
        super().__init__(G, agents)

    def _build_feature_vector(self, node: int, capability: str):
        agent = self.agents[node]
        neighbors = list(self.G.neighbors(node))
        degree_norm = self.G.degree[node] / max(1, len(self.G.nodes()))

        capable_neighbors = sum(
            1 for n in neighbors if capability in self.agents[n].capabilities
        )
        capable_neighbor_ratio = capable_neighbors / max(1, len(neighbors))
        neighbor_load = sum(self.agents[n].load for n in neighbors) / max(1, len(neighbors))
        avg_neighbor_reliability = sum(self.agents[n].reliability for n in neighbors) / max(1, len(neighbors))
        has_capability = 1.0 if capability in agent.capabilities else 0.0

        # Expanded feature vector
        return np.array(
            [
                1.0,  # bias
                agent.reliability,
                agent.resource,
                agent.load,
                degree_norm,
                has_capability,
                capable_neighbor_ratio,
                neighbor_load,
                avg_neighbor_reliability,  # new
                len(neighbors),           # new: local connectivity
            ],
            dtype=np.float64,
        )

    def train_guidance_model(self, train_tasks, feedback_tasks=None):
        """
        Train linear regression for guidance scores.
        Optional feedback_tasks can be used to slightly adjust weights
        based on real successes/failures.
        """
        super().compute_guidance()

        capability_frequency = {cap: 1.0 for cap in CAPABILITIES}
        for task in train_tasks:
            capability_frequency[task.required_capability] += 1.0

        for capability in CAPABILITIES:
            X_rows = []
            y_rows = []
            sample_weights = []

            for node in self.G.nodes():
                X_rows.append(self._build_feature_vector(node, capability))
                y_rows.append(self.guidance[node][capability])
                sample_weights.append(capability_frequency[capability])

            X = np.vstack(X_rows)
            y = np.array(y_rows, dtype=np.float64)
            w = np.array(sample_weights, dtype=np.float64)

            # Optional feedback: adjust targets slightly based on task success
            if feedback_tasks:
                for task in feedback_tasks:
                    if task.required_capability != capability:
                        continue
                    node = task.assigned_agent
                    if node in self.G.nodes():
                        success = 1.0 if task.success else 0.0
                        y[node] = 0.7 * y[node] + 0.3 * success  # blend heuristic with outcome

            sqrt_w = np.sqrt(w).reshape(-1, 1)
            Xw = X * sqrt_w
            yw = y * sqrt_w.flatten()

            # Linear regression weights (Moore-Penrose pseudo-inverse)
            weights = np.linalg.pinv(Xw.T @ Xw) @ (Xw.T @ yw)
            self.capability_weights[capability] = weights

        self.is_trained = True
        self.compute_guidance()

    def compute_guidance(self):
        super().compute_guidance()

        if not self.is_trained:
            return

        # Dynamic blending based on network congestion or task criticality
        avg_load = np.mean([agent.load for agent in self.agents.values()])
        self.dynamic_alpha = min(max(0.3, self.blend_alpha + (avg_load - 0.5)), 0.9)

        for node in self.G.nodes():
            for capability in CAPABILITIES:
                baseline = self.guidance[node][capability]
                features = self._build_feature_vector(node, capability)
                predicted = float(features @ self.capability_weights[capability])
                predicted = max(0.0, min(1.0, predicted))
                self.guidance[node][capability] = (
                    (1.0 - self.dynamic_alpha) * baseline
                    + self.dynamic_alpha * predicted
                )

    def explain_guidance(self, node: int, capability: str) -> str:
        base = super().explain_guidance(node, capability)
        return f"{base}, guidance_source=learned_blend, blend_alpha={self.dynamic_alpha:.2f}"
    def save(self, path: str):
        """
        Save coordinator checkpoint including learned weights and state.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "blend_alpha": np.array([self.blend_alpha], dtype=np.float64),
            "dynamic_alpha": np.array([self.dynamic_alpha], dtype=np.float64),
            "is_trained": np.array([1 if self.is_trained else 0], dtype=np.int32),
            "update_round": np.array([self.update_round], dtype=np.int32),
            "feature_schema_version": np.array([self.feature_schema_version]),
            "feature_names": np.array(self.feature_names),
        }
        for capability in CAPABILITIES:
            weights = self.capability_weights.get(capability)
            if weights is None:
                weights = np.zeros(10, dtype=np.float64)  # updated feature vector size
            payload[f"weights__{capability}"] = np.asarray(weights, dtype=np.float64)
        np.savez(path, **payload)

    def load(self, path: str):
        """
        Load coordinator checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Coordinator checkpoint not found: {path}")

        checkpoint = np.load(path, allow_pickle=False)
        self.blend_alpha = float(checkpoint["blend_alpha"][0])
        self.dynamic_alpha = float(np.atleast_1d(checkpoint.get("dynamic_alpha", np.array([self.blend_alpha])))[0])
        self.is_trained = bool(int(checkpoint["is_trained"][0]))
        self.update_round = int(checkpoint["update_round"][0])
        saved_feature_names = list(checkpoint["feature_names"])
        if saved_feature_names != self.feature_names:
            raise ValueError(
                "Coordinator feature schema mismatch. "
                "Delete outputs/models/latest/learned_global_coordinator.npz and retrain."
            )

        self.capability_weights = {}
        for capability in CAPABILITIES:
            key = f"weights__{capability}"
            if key in checkpoint:
                self.capability_weights[capability] = checkpoint[key]

        self.compute_guidance()
        return self