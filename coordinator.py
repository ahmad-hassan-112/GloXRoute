"""
coordinator.py — GlobalCoordinator with lazy on-demand guidance.

Key change: guidance is computed lazily per (node, capability) pair
only when first requested, rather than precomputing all 100×8=800 pairs
at initialisation. This makes the coordinator overhead proportional to
the routing decisions actually made, not to the full cross-product.

Overhead accounting (for paper):
  - Per routing step: one shortest-path lookup (amortised over cache hits)
  - State advertisement: one summary message per update round
  - Both are now measured and reported in experiment.py
"""

import networkx as nx
from typing import Dict, Optional
import numpy as np
import os

from config import CAPABILITIES
from models import Agent
from utils import capable_nodes, effective_edge_weight


class GlobalCoordinator:
    """
    Lightweight heuristic global coordinator.

    Provides per-(node, capability):
      - guidance score  in [0, 1]
      - recommended next-hop node

    Guidance is computed LAZILY: only when first requested for a given
    (node, capability) pair. Results are cached until update_guidance()
    is called (which clears the cache).
    """

    def __init__(self, G: nx.Graph, agents: Dict[int, Agent]):
        self.G = G
        self.agents = agents
        self._guidance_cache: Dict[tuple, float] = {}
        self._nexthop_cache: Dict[tuple, Optional[int]] = {}
        self._distance_cache: Dict[tuple, float] = {}
        self._dijkstra_cache: Dict[str, tuple] = {}   # capability → (distances, paths)
        self.state_version = 0
        self.update_round = 0
        self.last_summary: Dict = {}
        # Track how many cache hits vs misses for overhead reporting
        self._cache_hits = 0
        self._cache_misses = 0

    # ------------------------------------------------------------------ #
    #  Internal: run multi-source Dijkstra once per capability            #
    # ------------------------------------------------------------------ #

    def _run_dijkstra(self, capability: str):
        """Run multi-source Dijkstra from all capable nodes for `capability`."""
        if capability in self._dijkstra_cache:
            return self._dijkstra_cache[capability]

        targets = [
            n for n in capable_nodes(self.agents, capability)
            if not getattr(self.agents[n], "failed", False)
        ]

        if not targets:
            self._dijkstra_cache[capability] = ({}, {})
            return {}, {}

        weight = effective_edge_weight(self.agents)
        try:
            distances, paths = nx.multi_source_dijkstra(
                self.G, sources=set(targets), weight=weight
            )
        except Exception:
            distances, paths = {}, {}

        self._dijkstra_cache[capability] = (distances, paths)
        return distances, paths

    # ------------------------------------------------------------------ #
    #  Public: lazy accessors                                              #
    # ------------------------------------------------------------------ #

    def _ensure(self, node: int, capability: str):
        key = (node, capability)
        if key in self._guidance_cache:
            self._cache_hits += 1
            return

        self._cache_misses += 1
        distances, paths = self._run_dijkstra(capability)

        if getattr(self.agents[node], "failed", False) or node not in distances:
            self._guidance_cache[key]  = 0.0
            self._nexthop_cache[key]   = None
            self._distance_cache[key]  = float("inf")
            return

        cost = float(distances[node])
        self._distance_cache[key]  = cost
        self._guidance_cache[key]  = 1.0 / (1.0 + cost)

        path_to_node = paths[node]           # source→node from Dijkstra sources
        path_from_node = list(reversed(path_to_node))
        self._nexthop_cache[key] = path_from_node[1] if len(path_from_node) > 1 else node

    # public accessors (used by routers)
    @property
    def guidance(self):
        """Returns a lazy-proxy dict: guidance[node][cap]."""
        return _NestedProxy(self._guidance_access)

    @property
    def next_hop(self):
        return _NestedProxy(self._nexthop_access)

    @property
    def distance(self):
        return _NestedProxy(self._distance_access)

    def _guidance_access(self, node, cap):
        self._ensure(node, cap)
        return self._guidance_cache.get((node, cap), 0.0)

    def _nexthop_access(self, node, cap):
        self._ensure(node, cap)
        return self._nexthop_cache.get((node, cap))

    def _distance_access(self, node, cap):
        self._ensure(node, cap)
        return self._distance_cache.get((node, cap), float("inf"))

    # ------------------------------------------------------------------ #
    #  Overhead measurement                                                #
    # ------------------------------------------------------------------ #

    def overhead_stats(self) -> Dict:
        """Return cache hit/miss stats for overhead reporting."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits":   self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_lookups": total,
            "hit_rate": self._cache_hits / max(1, total),
        }

    # ------------------------------------------------------------------ #
    #  Update / refresh                                                    #
    # ------------------------------------------------------------------ #

    def update_guidance(self):
        """Clear caches and refresh (call periodically during experiment)."""
        self._guidance_cache.clear()
        self._nexthop_cache.clear()
        self._distance_cache.clear()
        self._dijkstra_cache.clear()
        self._cache_hits   = 0
        self._cache_misses = 0
        self.update_round += 1
        self.last_summary  = self.network_summary()
        self.state_version += 1

    def network_summary(self) -> Dict:
        loads       = [a.load        for a in self.agents.values()]
        resources   = [a.resource    for a in self.agents.values()]
        reliability = [a.reliability for a in self.agents.values()]
        return {
            "avg_load":        sum(loads)       / max(1, len(loads)),
            "avg_resource":    sum(resources)   / max(1, len(resources)),
            "avg_reliability": sum(reliability) / max(1, len(reliability)),
        }

    def congestion_penalty(self, node: int) -> float:
        return self.agents[node].load

    def explain_guidance(self, node: int, capability: str) -> str:
        guidance = self._guidance_access(node, capability)
        next_h   = self._nexthop_access(node, capability)
        agent    = self.agents[node]
        return (
            f"update_round={self.update_round}, "
            f"global_hint={guidance:.3f}, "
            f"recommended_next_hop={next_h}, "
            f"load={agent.load:.3f}, "
            f"reliability={agent.reliability:.3f}, "
            f"resource={agent.resource:.3f}"
        )


# ------------------------------------------------------------------ #
#  Lazy proxy helper                                                   #
# ------------------------------------------------------------------ #

class _NestedProxy:
    """
    Allows coordinator.guidance[node][capability] syntax while keeping
    guidance computation lazy.
    """
    def __init__(self, accessor):
        self._accessor = accessor

    def get(self, node, default=None):
        return _InnerProxy(self._accessor, node)

    def __getitem__(self, node):
        return _InnerProxy(self._accessor, node)


class _InnerProxy:
    def __init__(self, accessor, node):
        self._accessor = accessor
        self._node     = node

    def get(self, cap, default=None):
        try:
            return self._accessor(self._node, cap)
        except Exception:
            return default

    def __getitem__(self, cap):
        return self._accessor(self._node, cap)


# ------------------------------------------------------------------ #
#  LearnedGlobalCoordinator (kept for ablation completeness)          #
# ------------------------------------------------------------------ #

class LearnedGlobalCoordinator(GlobalCoordinator):
    """
    Extends GlobalCoordinator with a lightweight learned guidance model.
    Used only in the ablation study, not the main comparison.
    """

    def __init__(self, G: nx.Graph, agents: Dict[int, Agent], blend_alpha: float = 0.7):
        self.blend_alpha      = blend_alpha
        self.capability_weights: Dict[str, np.ndarray] = {}
        self.is_trained       = False
        self.dynamic_alpha    = blend_alpha
        self.feature_names    = [
            "bias", "reliability", "resource", "load",
            "degree_norm", "has_capability",
            "capable_neighbor_ratio", "neighbor_load",
            "avg_neighbor_reliability", "num_neighbors",
        ]
        self.feature_schema_version = "coord_v1"
        super().__init__(G, agents)

    def _build_feature_vector(self, node: int, capability: str) -> np.ndarray:
        agent     = self.agents[node]
        neighbors = list(self.G.neighbors(node))
        degree_norm = self.G.degree[node] / max(1, len(self.G.nodes()))
        capable_neighbors = sum(
            1 for n in neighbors if capability in self.agents[n].capabilities
        )
        capable_neighbor_ratio = capable_neighbors / max(1, len(neighbors))
        neighbor_load          = sum(self.agents[n].load for n in neighbors) / max(1, len(neighbors))
        avg_neighbor_rel       = sum(self.agents[n].reliability for n in neighbors) / max(1, len(neighbors))
        has_capability         = 1.0 if capability in agent.capabilities else 0.0
        return np.array([
            1.0, agent.reliability, agent.resource, agent.load,
            degree_norm, has_capability, capable_neighbor_ratio,
            neighbor_load, avg_neighbor_rel, float(len(neighbors)),
        ], dtype=np.float64)

    def train_guidance_model(self, train_tasks):
        # Build heuristic baseline first (populates Dijkstra cache)
        for cap in CAPABILITIES:
            self._run_dijkstra(cap)
            for node in self.G.nodes():
                self._ensure(node, cap)

        capability_frequency = {cap: 1.0 for cap in CAPABILITIES}
        for task in train_tasks:
            capability_frequency[task.required_capability] += 1.0

        for capability in CAPABILITIES:
            X_rows, y_rows, sample_weights = [], [], []
            for node in self.G.nodes():
                X_rows.append(self._build_feature_vector(node, capability))
                y_rows.append(self._guidance_cache.get((node, capability), 0.0))
                sample_weights.append(capability_frequency[capability])

            X = np.vstack(X_rows)
            y = np.array(y_rows, dtype=np.float64)
            w = np.sqrt(np.array(sample_weights, dtype=np.float64)).reshape(-1, 1)
            Xw, yw = X * w, y * w.flatten()
            weights = np.linalg.pinv(Xw.T @ Xw) @ (Xw.T @ yw)
            self.capability_weights[capability] = weights

        self.is_trained = True
        # Invalidate cache so learned scores are used on next lookup
        self.update_guidance()

    def _guidance_access(self, node, cap):
        """Override: blend heuristic + learned."""
        self._ensure(node, cap)
        baseline = self._guidance_cache.get((node, cap), 0.0)
        if not self.is_trained or cap not in self.capability_weights:
            return baseline
        features  = self._build_feature_vector(node, cap)
        predicted = float(np.clip(features @ self.capability_weights[cap], 0.0, 1.0))
        avg_load  = np.mean([a.load for a in self.agents.values()])
        alpha     = float(np.clip(self.blend_alpha + (avg_load - 0.5), 0.3, 0.9))
        return (1.0 - alpha) * baseline + alpha * predicted

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "blend_alpha":            np.array([self.blend_alpha]),
            "dynamic_alpha":          np.array([self.dynamic_alpha]),
            "is_trained":             np.array([1 if self.is_trained else 0], dtype=np.int32),
            "update_round":           np.array([self.update_round], dtype=np.int32),
            "feature_schema_version": np.array([self.feature_schema_version]),
            "feature_names":          np.array(self.feature_names),
        }
        for cap in CAPABILITIES:
            w = self.capability_weights.get(cap, np.zeros(10))
            payload[f"weights__{cap}"] = np.asarray(w, dtype=np.float64)
        np.savez(path, **payload)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Coordinator checkpoint not found: {path}")
        ckpt = np.load(path, allow_pickle=False)
        self.blend_alpha   = float(ckpt["blend_alpha"][0])
        self.dynamic_alpha = float(np.atleast_1d(ckpt.get("dynamic_alpha", np.array([self.blend_alpha])))[0])
        self.is_trained    = bool(int(ckpt["is_trained"][0]))
        self.update_round  = int(ckpt["update_round"][0])
        saved_names = list(ckpt["feature_names"])
        if saved_names != self.feature_names:
            raise ValueError(
                "Coordinator feature schema mismatch. Delete checkpoint and retrain."
            )
        self.capability_weights = {}
        for cap in CAPABILITIES:
            key = f"weights__{cap}"
            if key in ckpt:
                self.capability_weights[cap] = ckpt[key]
        self.update_guidance()
        return self