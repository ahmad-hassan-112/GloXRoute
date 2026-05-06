import networkx as nx
from typing import Dict, List
from models import Agent


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