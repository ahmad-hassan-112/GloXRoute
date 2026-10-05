"""
learned_routers.py — recent CCF-A learned baselines, adapted to the GloXRoute simulator.

DRLORSRouter  ("DRLOR_S")
    Adapted from:
      C. Liu, P. Wu, M. Xu, Y. Yang, N. Geng, "Scalable Deep Reinforcement
      Learning-Based Online Routing for Multi-Type Service Requirements",
      IEEE TPDS 34(8):2337-2351, 2023.  doi:10.1109/TPDS.2023.3284651
      (extends DRL-OR, IEEE INFOCOM 2021, doi:10.1109/INFOCOM42981.2021.9488736)

    What is kept from the original:
      * hop-by-hop routing by multi-agent DRL, one shared policy at every node
      * actor-critic training with a comprehensive reward (path cost, delivery,
        deadline / service requirement)
      * graph-aware actor: each candidate is scored from its own state plus an
        aggregation over its neighbourhood (a one-layer message-passing step,
        standing in for the GAT-based actor of DRL-OR-S)
      * link-state knowledge (distance to the nearest capable agent), as an
        SDN routing agent has
    What is adapted:
      * the action is "next agent" towards any agent holding the required
        capability, instead of "next router" towards a fixed destination
      * the policy is a small NumPy MLP (no torch dependency)

AgentNetStyleRouter  ("AgentNetStyle")
    Adapted from:
      Y. Yang, H. Chai, S. Shao, Y. Song, S. Qi, R. Rui, W. Zhang,
      "AgentNet: Decentralized Evolutionary Coordination for LLM-based
      Multi-Agent Systems", NeurIPS 2025.  arXiv:2504.00587

    What is kept from the original:
      * fully decentralised: no coordinator, no global state
      * every agent routes from its own retrieval memory of past tasks
      * agents' connection weights evolve with collaboration outcomes and
        weak connections are pruned
      * bounded memory with eviction of the least useful entries
      * learning continues online while tasks are served
    What is adapted:
      * no LLMs: retrieval is over structured task records (required
        capability -> next agent -> outcome), not text embeddings
      * the physical topology is fixed; "evolving connectivity" is a per-agent
        preference weight over existing links

Both routers produce NO explanations (they are black-box / memory-based),
which is the contrast GloXRoute's interpretability claim is made against.
"""

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from core.config import (
    MAX_HOPS, CAPABILITIES,
    DRLOR_TRAIN_MODE, DRLOR_TRAIN_EPISODES, DRLOR_TRAIN_GRAPHS, DRLOR_TRAIN_SEED_OFFSET,
    DRLOR_HIDDEN, DRLOR_LR, DRLOR_ENTROPY, DRLOR_BATCH,
    AGENTNET_WARMUP_TASKS, AGENTNET_MEMORY_SIZE, AGENTNET_PRUNE_THRESHOLD,
)
from core.models import RouteDecision
from core.utils import capable_nodes, source_can_solve, effective_edge_weight


# ====================================================================== #
#  Shared helpers                                                         #
# ====================================================================== #

def _alive(agents, n) -> bool:
    return not getattr(agents[n], "failed", False)


def _candidates(G, agents, current, visited, pruned=None) -> List[int]:
    cands = [n for n in G.neighbors(current) if n not in visited and _alive(agents, n)]
    if pruned:
        kept = [n for n in cands if n not in pruned]
        if kept:                       # never prune an agent into a dead end
            return kept
    return cands


def _cap_distances(G, agents, cap) -> Dict[int, float]:
    """Effective distance from every node to the nearest live agent holding
    `cap` (one multi-source Dijkstra)."""
    targets = [t for t in capable_nodes(agents, cap) if _alive(agents, t)]
    if not targets:
        return {}
    try:
        return nx.multi_source_dijkstra_path_length(G, set(targets),
                                                   weight=effective_edge_weight(agents))
    except Exception:
        return {}


def _cost_to_capable(G, agents) -> Dict[str, Dict[int, float]]:
    """Full link-state table (used for training environments)."""
    return {cap: _cap_distances(G, agents, cap) for cap in CAPABILITIES}


def _path_eff_cost(G, agents, path) -> float:
    w = effective_edge_weight(agents)
    tot = 0.0
    for u, v in zip(path[:-1], path[1:]):
        c = w(u, v, G[u][v])
        if c is None:
            return float("inf")
        tot += float(c)
    return tot


# ====================================================================== #
#  DRL-OR-S style router                                                  #
# ====================================================================== #

N_FEAT = 12


def _drl_features(G, agents, dist, current, n, required, visited, priority, hop):
    """Per-candidate state vector. All features are size-independent so a
    policy trained on one graph transfers to others (the DRL-OR-S goal)."""
    w = effective_edge_weight(agents)
    step = w(current, n, G[current][n])
    step = 50.0 if step is None else float(step)

    dc = dist.get(current, float("inf"))
    dn = dist.get(n, float("inf"))
    if math.isinf(dc) or dc <= 0:
        dc = 1.0
    reach = 0.0 if math.isinf(dn) else 1.0
    dn_f = 3.0 if math.isinf(dn) else dn

    nbrs = [x for x in G.neighbors(n) if x != current]
    alive_nbrs = [x for x in nbrs if _alive(agents, x)]
    onward = [x for x in alive_nbrs if x not in visited]
    k = max(1, len(alive_nbrs))
    nbr_load = sum(agents[x].load for x in alive_nbrs) / k
    nbr_cap = sum(1 for x in alive_nbrs if required in agents[x].capabilities) / k

    a = agents[n]
    return np.array([
        1.0 if required in a.capabilities else 0.0,          # delivers here
        (step + dn_f) / dc,                                    # progress ratio (1 = on a shortest path)
        step / 10.0,                                           # this hop's effective cost
        reach,                                                 # capable agent reachable from n
        a.load,
        a.reliability,
        a.resource,
        1.0 if agents[current].protocol == a.protocol else 0.0,
        min(len(onward), 10) / 10.0,                           # onward options (dead-end signal)
        nbr_load,                                              # message passing: neighbourhood load
        nbr_cap,                                               # message passing: capable neighbours
        priority,                                              # service type (tight vs loose deadline)
    ], dtype=float)


class _PolicyNet:
    """Shared actor (per-candidate MLP scorer, softmax over candidates)
    and critic (linear value on state summary), trained with Adam."""

    def __init__(self, n_in, hidden, rng):
        s = 1.0 / math.sqrt(n_in)
        self.W1 = rng.normal(0, s, (hidden, n_in))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1.0 / math.sqrt(hidden), hidden)
        self.v = np.zeros(4)               # critic weights
        self.params = ["W1", "b1", "w2"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.s = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def logits(self, X):
        H = np.tanh(X @ self.W1.T + self.b1)
        return H @ self.w2, H

    def value(self, z):
        return float(self.v @ z)

    def adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.s[p] = b2 * self.s[p] + (1 - b2) * g * g
            mh = self.m[p] / (1 - b1 ** self.t)
            sh = self.s[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(sh) + eps))

    def to_dict(self):
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2, "v": self.v}


# Trained policies are shared across seeds of the same configuration in
# "transfer" mode, so each (topology, condition) is trained exactly once.
_POLICY_CACHE: Dict[tuple, Tuple[_PolicyNet, dict]] = {}


class DRLORSRouter:
    name = "DRLOR_S"

    def __init__(self, topology: str, num_agents: int, eval_seed: int,
                 train_mode: str = DRLOR_TRAIN_MODE):
        self.topology = topology
        self.num_agents = num_agents
        self.eval_seed = eval_seed
        self.train_mode = train_mode
        self.net: Optional[_PolicyNet] = None
        self.train_stats: dict = {}
        # Link-state table, computed lazily per capability and cleared by
        # refresh_state() on the same schedule as GloXRoute's coordinator.
        self._dist: Dict[str, Dict[int, float]] = {}
        self._G = None
        # control messages: |V| per table computation, the same charge as one
        # coordinator Dijkstra run (see experiment._coordinator_overhead_per_task)
        self.control_messages_total = 0.0

    def refresh_state(self):
        self._dist = {}

    def _table(self, G, agents, cap):
        if cap not in self._dist:
            self._dist[cap] = _cap_distances(G, agents, cap)
            self.control_messages_total += G.number_of_nodes()
        return self._dist[cap]

    # ------------------------------------------------------------------ #
    def prepare(self, G, agents):
        """Train (or fetch the cached policy) before evaluation."""
        from core import config as cfg
        key = (self.topology, self.num_agents, cfg.CONGESTION_SCENARIO,
               round(cfg.FAILURE_RATE, 4), self.train_mode,
               self.eval_seed if self.train_mode == "same_graph" else None)
        if key in _POLICY_CACHE:
            self.net, stats = _POLICY_CACHE[key]
            self.train_stats = dict(stats, reused=True)
            return

        # keep the experiment's global RNG untouched
        py_state, np_state = random.getstate(), np.random.get_state()
        t0 = time.perf_counter()
        rng = np.random.default_rng(10_000 + self.eval_seed)

        if self.train_mode == "same_graph":
            envs = [(G, agents)]
        else:
            from core.network_builder import build_ioa_network
            envs = [build_ioa_network(seed=DRLOR_TRAIN_SEED_OFFSET + i,
                                      topology=self.topology,
                                      num_agents=self.num_agents)
                    for i in range(DRLOR_TRAIN_GRAPHS)]

        self.net = _PolicyNet(N_FEAT, DRLOR_HIDDEN, rng)
        curve = self._train(envs, rng)

        random.setstate(py_state)
        np.random.set_state(np_state)
        self.train_stats = {
            "train_mode": self.train_mode,
            "train_graphs": len(envs),
            "train_episodes": DRLOR_TRAIN_EPISODES,
            "train_time_s": time.perf_counter() - t0,
            "final_success_rate": curve[-1][0] if curve else float("nan"),
            "final_mean_stretch": curve[-1][1] if curve else float("nan"),
            "reused": False,
        }
        _POLICY_CACHE[key] = (self.net, self.train_stats)

    # ------------------------------------------------------------------ #
    def _episode(self, G, agents, dist_tab, source, required, priority, deadline,
                 rng, greedy):
        dist = dist_tab(required) if callable(dist_tab) else dist_tab.get(required, {})
        current, path, visited = source, [source], {source}
        traj = []   # (X, action_idx, critic_input, step_reward)
        best = dist.get(source, float("inf"))
        w = effective_edge_weight(agents)

        for hop in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            cands = _candidates(G, agents, current, visited)
            if not cands:
                break
            X = np.stack([_drl_features(G, agents, dist, current, n, required,
                                        visited, priority, hop) for n in cands])
            z, _ = self.net.logits(X)
            if greedy:
                a = int(np.argmax(z))
            else:
                p = np.exp(z - z.max()); p /= p.sum()
                a = int(rng.choice(len(cands), p=p))
            n = cands[a]
            c = w(current, n, G[current][n])
            step = 50.0 if c is None else float(c)
            dc = dist.get(current, float("inf"))
            crit = np.array([1.0, 0.0 if math.isinf(dc) else dc / max(best, 1e-6),
                             hop / MAX_HOPS, priority])
            norm = best if (best > 0 and not math.isinf(best)) else 10.0
            traj.append((X, a, crit, -step / norm))
            current = n
            path.append(n)
            visited.add(n)

        delivered = required in agents[current].capabilities
        cost = _path_eff_cost(G, agents, path)
        terminal = 1.0 if delivered else -2.0
        if delivered and best > 0 and not math.isinf(best) and cost > deadline * best:
            terminal -= 1.0                                   # service requirement missed
        return path, traj, terminal, delivered, (cost / best if delivered and best > 0 else float("nan"))

    def _train(self, envs, rng):
        tables = [_cost_to_capable(G, A) for G, A in envs]
        grads = {p: np.zeros_like(getattr(self.net, p)) for p in self.net.params}
        v_grad = np.zeros_like(self.net.v)
        curve, win_s, win_str = [], [], []

        for ep in range(DRLOR_TRAIN_EPISODES):
            ei = int(rng.integers(len(envs)))
            G, A = envs[ei]
            nodes = [n for n in G.nodes() if _alive(A, n)]
            src = int(nodes[rng.integers(len(nodes))])
            req = CAPABILITIES[int(rng.integers(len(CAPABILITIES)))]
            if req in A[src].capabilities:
                continue
            pr = float(rng.uniform(0.5, 1.0))
            dl = 1.0 + (1.0 - pr) * 0.8
            _, traj, term, ok, stretch = self._episode(G, A, tables[ei], src, req, pr, dl,
                                                       rng, greedy=False)
            win_s.append(ok)
            if ok:
                win_str.append(stretch)
            if not traj:
                continue

            # returns-to-go (gamma = 1, short episodes)
            rews = [t[3] for t in traj]
            rews[-1] += term
            G_t = np.cumsum(rews[::-1])[::-1]

            for (X, a, crit, _), ret in zip(traj, G_t):
                V = self.net.value(crit)
                adv = ret - V
                v_grad += -(adv) * crit                        # d/dv of 0.5*(ret-V)^2

                z, H = self.net.logits(X)
                p = np.exp(z - z.max()); p /= p.sum()
                logp = np.log(p + 1e-12)
                ent = -(p * logp).sum()
                dz = -adv * (np.eye(len(p))[a] - p)            # policy-gradient loss
                dz += DRLOR_ENTROPY * p * (logp + ent)         # minus entropy bonus
                grads["w2"] += H.T @ dz
                dpre = (dz[:, None] * self.net.w2[None, :]) * (1 - H ** 2)
                grads["W1"] += dpre.T @ X
                grads["b1"] += dpre.sum(axis=0)

            if (ep + 1) % DRLOR_BATCH == 0:
                for p in grads:
                    grads[p] /= DRLOR_BATCH
                    np.clip(grads[p], -5, 5, out=grads[p])
                self.net.adam(grads, DRLOR_LR)
                self.net.v -= 0.01 * v_grad / DRLOR_BATCH
                grads = {p: np.zeros_like(getattr(self.net, p)) for p in self.net.params}
                v_grad[:] = 0

            if (ep + 1) % 500 == 0:
                curve.append((float(np.mean(win_s)),
                              float(np.nanmean(win_str)) if win_str else float("nan")))
                win_s, win_str = [], []
        return curve

    # ------------------------------------------------------------------ #
    def route(self, G, agents, task) -> RouteDecision:
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)
        if self._G is not G:
            self._G, self._dist, self.control_messages_total = G, {}, 0.0
        path, _, _, _, _ = self._episode(
            G, agents, lambda cap: self._table(G, agents, cap), task.source, task.required_capability,
            float(task.priority), float(task.deadline), None, greedy=True)
        return RouteDecision(path=path, explanation="", extra_overhead=0.0)


# ====================================================================== #
#  AgentNet style router                                                  #
# ====================================================================== #

class AgentNetStyleRouter:
    name = "AgentNetStyle"

    def __init__(self, eval_seed: int, warmup_tasks: int = AGENTNET_WARMUP_TASKS,
                 memory_size: int = AGENTNET_MEMORY_SIZE,
                 prune_threshold: float = AGENTNET_PRUNE_THRESHOLD,
                 explore_c: float = 0.5, lr: float = 0.1):
        self.eval_seed = eval_seed
        self.warmup_tasks = warmup_tasks
        self.memory_size = memory_size
        self.prune_threshold = prune_threshold
        self.explore_c = explore_c
        self.lr = lr
        # memory[agent][(capability, neighbour)] = [count, successes, mean_cost_to_go, last_used]
        self.memory: Dict[int, Dict[tuple, list]] = defaultdict(dict)
        # evolving connection weights: link_w[agent][neighbour]
        self.link_w: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.clock = 0
        self.train_stats: dict = {}
        self._explore = True

    # ---------------------------- retrieval ----------------------------- #
    def _retrieve(self, a, required, n):
        """Exact record for (capability, neighbour); otherwise aggregate the
        neighbour's records for other capabilities (weaker evidence)."""
        mem = self.memory[a]
        rec = mem.get((required, n))
        if rec is not None:
            return rec[0], rec[1] / rec[0], rec[2], 1.0
        others = [r for (c, m), r in mem.items() if m == n]
        if others:
            cnt = sum(r[0] for r in others)
            succ = sum(r[1] for r in others) / cnt
            cost = sum(r[2] * r[0] for r in others) / cnt
            return cnt, succ, cost, 0.3
        return 0, None, None, 0.0

    def _score(self, G, agents, a, n, required):
        ag = agents[n]
        # self-description the neighbour advertises (AgentNet agents know
        # their neighbours' declared abilities)
        prior = (2.0 * (required in ag.capabilities)
                 + 0.5 * ag.reliability - 0.5 * ag.load
                 - 0.05 * G[a][n]["latency"])
        cnt, succ, cost, rel = self._retrieve(a, required, n)
        score = prior
        if cnt:
            score += rel * (2.0 * succ - 0.1 * cost)
        score += 0.3 * math.log(self.link_w[a][n] + 1e-6)
        if self._explore:
            score += self.explore_c * math.sqrt(math.log(self.clock + 2) / (1 + cnt))
        return score

    def _pruned(self, a):
        w = self.link_w.get(a)
        if not w:
            return None
        return {n for n, v in w.items() if v < self.prune_threshold}

    def _walk(self, G, agents, source, required):
        current, path, visited = source, [source], {source}
        for _ in range(MAX_HOPS):
            if required in agents[current].capabilities:
                break
            cands = _candidates(G, agents, current, visited, self._pruned(current))
            if not cands:
                break
            n = max(cands, key=lambda m: self._score(G, agents, current, m, required))
            path.append(n)
            visited.add(n)
            current = n
        return path

    # ----------------------------- learning ----------------------------- #
    def feedback(self, G, agents, task, path):
        """Credit every agent on the path with the outcome of its choice."""
        self.clock += 1
        if len(path) < 2:
            return
        required = task.required_capability
        ok = required in agents[path[-1]].capabilities
        w = effective_edge_weight(agents)
        costs = []
        for u, v in zip(path[:-1], path[1:]):
            c = w(u, v, G[u][v])
            costs.append(50.0 if c is None else float(c))
        to_go = np.cumsum(costs[::-1])[::-1]
        for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
            mem = self.memory[u]
            rec = mem.get((required, v))
            if rec is None:
                rec = [0, 0, 0.0, self.clock]
                mem[(required, v)] = rec
            rec[0] += 1
            rec[1] += int(ok)
            rec[2] += (float(to_go[i]) - rec[2]) / rec[0]
            rec[3] = self.clock
            # connection evolution
            lw = self.link_w[u]
            lw[v] = max(0.05, min(3.0, lw[v] * (1 + self.lr) if ok else lw[v] * (1 - self.lr)))
            # bounded memory: evict the least used / oldest record
            if len(mem) > self.memory_size:
                victim = min(mem, key=lambda k: (mem[k][0], mem[k][3]))
                del mem[victim]

    def prepare(self, G, agents):
        """Warm-up: the network serves a stream of tasks before evaluation so
        agents have populated memories (AgentNet's evolution phase)."""
        t0 = time.perf_counter()
        rng = random.Random(20_000 + self.eval_seed)
        nodes = [n for n in G.nodes() if _alive(agents, n)]
        served = ok = 0
        from core.models import Task
        for i in range(self.warmup_tasks):
            src = rng.choice(nodes)
            req = rng.choice(CAPABILITIES)
            if req in agents[src].capabilities:
                continue
            t = Task(task_id=-1 - i, source=src, required_capability=req,
                     priority=1.0, deadline=1.0)
            path = self._walk(G, agents, src, req)
            self.feedback(G, agents, t, path)
            served += 1
            ok += int(req in agents[path[-1]].capabilities)
        self._explore = False                     # exploit during evaluation; keep learning online
        self.train_stats = {
            "warmup_tasks": served,
            "warmup_success_rate": ok / max(1, served),
            "train_time_s": time.perf_counter() - t0,
            "memory_records": sum(len(m) for m in self.memory.values()),
        }

    def route(self, G, agents, task) -> RouteDecision:
        if source_can_solve(agents, task):
            return RouteDecision(path=[task.source], explanation="", extra_overhead=0.0)
        path = self._walk(G, agents, task.source, task.required_capability)
        # outcome feedback travels back along the path: one message per hop
        return RouteDecision(path=path, explanation="", extra_overhead=float(len(path) - 1))
