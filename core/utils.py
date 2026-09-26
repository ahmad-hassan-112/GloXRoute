import networkx as nx
from typing import Dict, List
from core.models import Agent
import math
import networkx as nx

from core.config import PROTOCOL_MODE, PROTOCOL_PENALTY, CONGESTION_LAMBDA


def capable_nodes(agents: Dict[int, Agent], capability: str) -> List[int]:
    return [
        agent_id
        for agent_id, agent in agents.items()
        if capability in agent.capabilities
    ]


def path_latency(G: nx.Graph, agents: Dict[int, Agent], path: List[int]) -> float:
    if len(path) <= 1:
        return 0.0

    total_latency = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge_latency = G[u][v]["latency"]
        receiver_load = agents[v].load
        total_latency += edge_latency * (1.0 + receiver_load)

    return total_latency


# IMPROVEMENTS

def source_can_solve(agents, task):
    agent = agents[task.source]
    return (
        not getattr(agent, "failed", False)
        and task.required_capability in agent.capabilities
    )


def protocol_penalty(agents, u, v):
    if agents[u].protocol == agents[v].protocol:
        return 0.0
    return PROTOCOL_PENALTY


def effective_edge_weight(agents):
    """
    NetworkX weight function.
    If it returns None, NetworkX hides the edge.
    """
    def weight(u, v, data):
        if getattr(agents[u], "failed", False) or getattr(agents[v], "failed", False):
            return None

        base = float(data.get("latency", 1.0))

        if PROTOCOL_MODE == "hard_block":
            if agents[u].protocol != agents[v].protocol:
                return None

        if PROTOCOL_MODE == "soft_penalty":
            base += protocol_penalty(agents, u, v)

        base += CONGESTION_LAMBDA * max(agents[u].load, agents[v].load)

        return base

    return weight


def shortest_capability_latency(G, agents, task):
    targets = [
        node for node, agent in agents.items()
        if not getattr(agent, "failed", False)
        and task.required_capability in agent.capabilities
    ]

    if not targets:
        return math.inf

    if task.source in targets:
        return 0.0

    best = math.inf
    weight = effective_edge_weight(agents)

    for target in targets:
        try:
            d = nx.shortest_path_length(
                G,
                source=task.source,
                target=target,
                weight=weight,
            )
            best = min(best, float(d))
        except nx.NetworkXNoPath:
            continue

    return best

def path_effective_latency(G, agents, path):
    """
    Uses the same effective edge weight as the routers.
    This includes protocol penalty, load penalty, and failure blocking.
    """
    if path is None or len(path) <= 1:
        return 0.0

    weight_fn = effective_edge_weight(agents)
    total = 0.0

    for u, v in zip(path[:-1], path[1:]):
        if not G.has_edge(u, v):
            return float("inf")

        cost = weight_fn(u, v, G[u][v])

        if cost is None:
            return float("inf")

        total += float(cost)

    return total

