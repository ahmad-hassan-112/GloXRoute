"""
network_builder.py — build an IoA graph with heterogeneous agents.

Supported topologies
--------------------
  "erdos_renyi"      — original ER graph (p=0.06)
  "barabasi_albert"  — BA scale-free  (PRIMARY for paper)
  "watts_strogatz"   — WS small-world

Citation for BA: Barabási & Albert (1999). "Emergence of scaling in
random networks." Science, 286(5439), 509-512.
Citation for WS: Watts & Strogatz (1998). "Collective dynamics of
'small-world' networks." Nature, 393(6684), 440-442.
"""
import random
import numpy as np
import networkx as nx
from typing import Dict, Tuple

from config import (
    NUM_AGENTS, CAPABILITIES, PROTOCOLS,
    TOPOLOGY, BA_M, WS_K, WS_BETA,
    CAPS_PER_AGENT_MIN, CAPS_PER_AGENT_MAX,
    FAILURE_RATE, CONGESTION_SCENARIO, CONGESTION_FRACTION,
)
from models import Agent


# ------------------------------------------------------------------ #
#  Graph builders                                                      #
# ------------------------------------------------------------------ #

def _make_ba_graph(num_agents: int, seed: int) -> nx.Graph:
    """Barabási-Albert preferential-attachment graph."""
    G = nx.barabasi_albert_graph(n=num_agents, m=BA_M, seed=seed)
    assert nx.is_connected(G), "BA graph should always be connected"
    return G


def _make_ws_graph(num_agents: int, seed: int) -> nx.Graph:
    """Watts-Strogatz small-world graph; retry until connected."""
    rng = random.Random(seed)
    for _ in range(200):
        G = nx.watts_strogatz_graph(
            n=num_agents, k=WS_K, p=WS_BETA,
            seed=rng.randint(1, 10**6)
        )
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected WS graph in 200 tries.")


def _make_er_graph(num_agents: int, seed: int) -> nx.Graph:
    """Erdős-Rényi graph; retry until connected."""
    rng = random.Random(seed)
    for _ in range(200):
        G = nx.erdos_renyi_graph(
            n=num_agents, p=0.06,
            seed=rng.randint(1, 10**6)
        )
        if nx.is_connected(G):
            return G
    raise RuntimeError("Could not generate a connected ER graph in 200 tries.")


def create_network(
    num_agents: int = NUM_AGENTS,
    seed: int = 42,
    topology: str = TOPOLOGY,
) -> nx.Graph:
    random.seed(seed)
    np.random.seed(seed)

    if topology == "barabasi_albert":
        G = _make_ba_graph(num_agents, seed)
    elif topology == "watts_strogatz":
        G = _make_ws_graph(num_agents, seed)
    else:
        G = _make_er_graph(num_agents, seed)

    # Assign edge attributes
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["latency"]   = float(rng.uniform(1.0, 10.0))
        G[u][v]["bandwidth"] = float(rng.uniform(0.5, 5.0))

    return G


# ------------------------------------------------------------------ #
#  Agent builder                                                       #
# ------------------------------------------------------------------ #

def create_agents(G: nx.Graph, seed: int = 42) -> Dict[int, Agent]:
    rng = np.random.default_rng(seed + 1)
    random.seed(seed + 2)

    agents = {}
    for node in G.nodes():
        n_caps = int(rng.integers(CAPS_PER_AGENT_MIN, CAPS_PER_AGENT_MAX + 1))
        caps   = set(random.sample(CAPABILITIES, n_caps))

        # Under a congestion scenario, make CONGESTION_FRACTION of agents very busy
        if CONGESTION_SCENARIO and rng.random() < CONGESTION_FRACTION:
            load = float(rng.beta(5, 1))     # heavy-tail towards 1
        else:
            load = float(rng.uniform(0.0, 0.5))   # low load otherwise

        failed = FAILURE_RATE > 0 and rng.random() < FAILURE_RATE

        agents[node] = Agent(
            agent_id    = node,
            capabilities = caps,
            reliability = float(rng.uniform(0.65, 0.99)),
            load        = load,
            resource    = float(rng.uniform(0.4, 1.0)),
            protocol    = random.choice(PROTOCOLS),
            failed      = failed,
        )

    return agents


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def build_ioa_network(
    seed: int = 42,
    topology: str = TOPOLOGY,
    num_agents: int = NUM_AGENTS,
) -> Tuple[nx.Graph, Dict[int, Agent]]:
    G      = create_network(num_agents=num_agents, seed=seed, topology=topology)
    agents = create_agents(G, seed=seed)
    return G, agents