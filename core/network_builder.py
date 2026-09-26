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

from core.config import (
    NUM_AGENTS, CAPABILITIES, PROTOCOLS,
    TOPOLOGY, BA_M, WS_K, WS_BETA,
    CAPS_PER_AGENT_MIN, CAPS_PER_AGENT_MAX,
    FAILURE_RATE, CONGESTION_SCENARIO, CONGESTION_FRACTION,
    CAPABILITY_ZIPF_EXPONENT, MIN_AGENTS_PER_CAPABILITY,
)
from core.models import Agent


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

def _capability_quota(num_agents: int) -> dict:
    """
    Decide how many agents hold each capability, following a Zipf
    distribution so that some capabilities are genuinely SCARCE.

    With exponent=1.5 and 100 agents over 8 capabilities this yields
    roughly {52, 18, 10, 7, 5, 4, 3, 2} holders — a 26x range between
    the most common and rarest capability. Routing to a rare capability
    then requires real multi-hop search.
    """
    n_caps  = len(CAPABILITIES)
    weights = np.array(
        [1.0 / ((i + 1) ** CAPABILITY_ZIPF_EXPONENT) for i in range(n_caps)]
    )
    weights = weights / weights.sum()

    quota = np.floor(weights * num_agents).astype(int)
    quota = np.maximum(quota, MIN_AGENTS_PER_CAPABILITY)

    # print("🚀 ~ _capability_quota ~ quota:", quota)

    # Reconcile so total assignments == num_agents (1 capability per agent)
    diff = num_agents - int(quota.sum())
    i = 0
    while diff != 0:
        # print("🚀 ~ _capability_quota ~ diff:", diff)
        idx = i % n_caps
        if diff > 0:
            quota[idx] += 1
            diff -= 1
        elif quota[idx] > MIN_AGENTS_PER_CAPABILITY:
            quota[idx] -= 1
            diff += 1
        i += 1

    # print("🚀 ~ _capability_quota ~ {CAPABILITIES[i]: int(quota[i]) for i in range(n_caps)}:", {CAPABILITIES[i]: int(quota[i]) for i in range(n_caps)})
    return {CAPABILITIES[i]: int(quota[i]) for i in range(n_caps)}


def create_agents(G: nx.Graph, seed: int = 42) -> Dict[int, Agent]:
    rng = np.random.default_rng(seed + 1)
    random.seed(seed + 2)

    nodes = list(G.nodes())

    # --- Assign capabilities according to the scarcity quota -------------
    quota = _capability_quota(len(nodes))
    cap_pool = []
    for cap, count in quota.items():
        cap_pool.extend([cap] * count)
    rng.shuffle(cap_pool)
    # Guard against off-by-one
    while len(cap_pool) < len(nodes):
        cap_pool.append(CAPABILITIES[0])
    cap_pool = cap_pool[:len(nodes)]

    agents = {}
    for i, node in enumerate(nodes):
        caps = {cap_pool[i]}

        # Extra capabilities if configured (default: none)
        extra = int(rng.integers(CAPS_PER_AGENT_MIN, CAPS_PER_AGENT_MAX + 1)) - 1
        if extra > 0:
            caps |= set(random.sample(CAPABILITIES, extra))

        # Congestion scenario: a fraction of agents become heavily loaded
        if CONGESTION_SCENARIO and rng.random() < CONGESTION_FRACTION:
            load = float(rng.beta(5, 1))          # heavy tail toward 1
        else:
            load = float(rng.uniform(0.0, 0.5))

        failed = FAILURE_RATE > 0 and rng.random() < FAILURE_RATE

        agents[node] = Agent(
            agent_id     = node,
            capabilities = caps,
            reliability  = float(rng.uniform(0.65, 0.99)),
            load         = load,
            resource     = float(rng.uniform(0.4, 1.0)),
            protocol     = random.choice(PROTOCOLS),
            failed       = failed,
        )

    return agents


def capability_distribution(agents: Dict[int, Agent]) -> dict:
    """Report how many agents hold each capability (for the paper)."""
    counts = {c: 0 for c in CAPABILITIES}
    for a in agents.values():
        for c in a.capabilities:
            counts[c] = counts.get(c, 0) + 1
    return counts


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