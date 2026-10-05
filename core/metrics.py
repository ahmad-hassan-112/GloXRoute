from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import networkx as nx

from core.models import Agent, Task
from core.utils import path_effective_latency, shortest_capability_latency, effective_edge_weight


@dataclass
class RouteResult:
    method: str
    task_id: int
    success: bool
    latency: float
    hops: int
    communication_overhead: float
    explanation_available: bool
    explanation: str
    source_capable: bool

    # New routing-quality metrics
    shortest_possible_latency: float
    path_stretch: float
    loop_detected: bool

    # New congestion / load metrics
    avg_load_on_path: float
    max_load_on_path: float

    # New heterogeneity / protocol metrics
    protocol_mismatch_rate: float

    # New XAI metrics
    explanation_length: int

    # Deadline / QoS metric (priority-scaled)
    deadline_miss: bool


def _safe_latency(G: nx.Graph, path: List[int]) -> float:
    if path is None or len(path) <= 1:
        return 0.0

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            return float("inf")
        total += float(G[u][v].get("latency", 1.0))
    return total


def _has_required_capability(agents: Dict[int, Agent], node: int, capability: str) -> bool:
    if node not in agents:
        return False
    return capability in agents[node].capabilities


def _shortest_possible_latency(G: nx.Graph, agents: Dict[int, Agent], task: Task) -> float:
    targets = [
        node for node, agent in agents.items()
        if task.required_capability in agent.capabilities
    ]

    if not targets:
        return float("inf")

    best = float("inf")

    for target in targets:
        try:
            d = nx.shortest_path_length(
                G,
                source=task.source,
                target=target,
                weight="latency"
            )
            best = min(best, float(d))
        except nx.NetworkXNoPath:
            continue

    return best


def _avg_load_on_path(agents: Dict[int, Agent], path: List[int]) -> float:
    if not path:
        return 0.0
    loads = [float(agents[node].load) for node in path if node in agents]
    if not loads:
        return 0.0
    return sum(loads) / len(loads)


def _max_load_on_path(agents: Dict[int, Agent], path: List[int]) -> float:
    if not path:
        return 0.0
    loads = [float(agents[node].load) for node in path if node in agents]
    if not loads:
        return 0.0
    return max(loads)


def _protocol_mismatch_rate(agents: Dict[int, Agent], path: List[int]) -> float:
    if path is None or len(path) <= 1:
        return 0.0

    mismatches = 0
    edges = 0

    for u, v in zip(path[:-1], path[1:]):
        if u in agents and v in agents:
            edges += 1
            if agents[u].protocol != agents[v].protocol:
                mismatches += 1

    if edges == 0:
        return 0.0

    return mismatches / edges


def _explanation_length(explanation: Optional[str]) -> int:
    if not explanation:
        return 0
    return len(str(explanation).split())


# ------------------------------------------------------------------ #
#  Stochastic delivery delay (common random numbers)                   #
# ------------------------------------------------------------------ #
_JITTER_SEED = 0


def set_jitter_seed(seed: int) -> None:
    """Called once per seed by every experiment runner."""
    global _JITTER_SEED
    _JITTER_SEED = int(seed)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def _uniform(*keys) -> float:
    h = 0
    for k in keys:
        h = _splitmix64(h ^ (int(k) & 0xFFFFFFFFFFFFFFFF))
    return ((h >> 11) + 0.5) / float(1 << 53)          # in (0, 1)


def jitter_factor(task_id: int, hop: int, u: int, v: int) -> float:
    """Mean-1 log-normal delay factor for traversing (u, v) as hop `hop` of
    task `task_id`. Identical for every method taking that hop."""
    from core.config import DELAY_JITTER_CV as cv
    if cv <= 0:
        return 1.0
    u1 = _uniform(_JITTER_SEED, task_id, hop, u, v, 1)
    u2 = _uniform(_JITTER_SEED, task_id, hop, u, v, 2)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    s2 = math.log(1.0 + cv * cv)
    return math.exp(-0.5 * s2 + math.sqrt(s2) * z)


def realised_latency(G, agents, task, path) -> float:
    """Actual delivery delay of a route: expected hop costs x jitter."""
    if path is None or len(path) <= 1:
        return 0.0
    w = effective_edge_weight(agents)
    total = 0.0
    for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
        if not G.has_edge(u, v):
            return float("inf")
        c = w(u, v, G[u][v])
        if c is None:
            return float("inf")
        total += float(c) * jitter_factor(task.task_id, i, u, v)
    return total


def evaluate_route(
    G: nx.Graph,
    agents: Dict[int, Agent],
    task: Task,
    method_name: str,
    path: List[int],
    extra_overhead: float = 0.0,
    explanation: Optional[str] = None,
    decision_time: float = 0.0,
):
    if path is None:
        path = []

    latency = _safe_latency(G, path)
    effective_latency = path_effective_latency(G, agents, path)
    hops = max(0, len(path) - 1)

    valid_path = len(path) > 0 and path[0] == task.source
    reached_capable_agent = (
        valid_path
        and len(path) > 0
        and _has_required_capability(agents, path[-1], task.required_capability)
    )

    # A route that traverses a failed/blocked edge is NOT a success, even if
    # the final node has the capability. Previously such routes were counted
    # as successful while reporting inf effective latency, which both inflated
    # success_rate and destroyed the effective-latency mean.
    route_is_traversable = not math.isinf(effective_latency)

    success = route_is_traversable and (
        valid_path
        and reached_capable_agent
        and latency != float("inf")
    )

    shortest_latency = _shortest_possible_latency(G, agents, task)

    if shortest_latency == 0.0:
        path_stretch = 1.0 if hops == 0 else None
    elif math.isinf(shortest_latency) or math.isinf(latency):
        path_stretch = None
    else:
        path_stretch = latency / shortest_latency

    source_capable = shortest_latency == 0.0

    loop_detected = len(path) != len(set(path))

    communication_overhead = hops + float(extra_overhead)

    # Deadline miss: the REALISED delay (with delivery jitter) exceeds
    # deadline_factor x best achievable EXPECTED latency.
    best_effective = shortest_capability_latency(G, agents, task)
    realised = realised_latency(G, agents, task, path)
    if math.isinf(best_effective) or math.isinf(realised):
        deadline_miss = True
    elif best_effective <= 0.0:
        deadline_miss = False
    else:
        deadline_miss = realised > (float(task.deadline) * best_effective)

    result = RouteResult(
        method=method_name,
        task_id=task.task_id,
        success=success,
        latency=latency,
        hops=hops,
        communication_overhead=communication_overhead,
        explanation_available=bool(explanation),

        shortest_possible_latency=shortest_latency,
        path_stretch=path_stretch,
        loop_detected=loop_detected,

        avg_load_on_path=_avg_load_on_path(agents, path),
        max_load_on_path=_max_load_on_path(agents, path),
        protocol_mismatch_rate=_protocol_mismatch_rate(agents, path),
        explanation_length=_explanation_length(explanation),
        explanation=str(explanation) if explanation else "",
        source_capable=source_capable,
        deadline_miss=deadline_miss,
    )

    row = result.__dict__
    row["decision_time_ms"] = decision_time * 1000.0
    row["effective_latency"] = effective_latency
    row["realised_latency"]  = realised

    return row