import random
import numpy as np
import networkx as nx
from typing import Dict, Tuple

from config import NUM_AGENTS, CAPABILITIES, PROTOCOLS
from models import Agent


def create_network(num_agents: int = NUM_AGENTS, seed: int = 42) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)

    while True:
        G = nx.erdos_renyi_graph(
            n=num_agents,
            p=0.06,
            seed=random.randint(1, 999999)
        )

        if nx.is_connected(G):
            break

    for u, v in G.edges():
        G[u][v]["latency"] = random.uniform(1.0, 10.0)
        G[u][v]["bandwidth"] = random.uniform(0.5, 5.0)

    return G


def create_agents(G: nx.Graph) -> Dict[int, Agent]:
    agents = {}

    for node in G.nodes():
        agents[node] = Agent(
            agent_id=node,
            capabilities=set(random.sample(CAPABILITIES, 1)),
            reliability=random.uniform(0.65, 0.99),
            load=random.uniform(0.0, 1.0),
            resource=random.uniform(0.4, 1.0),
            protocol=random.choice(PROTOCOLS),
        )

    return agents


def build_ioa_network(seed: int = 42) -> Tuple[nx.Graph, Dict[int, Agent]]:
    G = create_network(seed=seed)
    agents = create_agents(G)
    return G, agents