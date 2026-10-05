"""
dynamics.py — time-varying agent load (DYNAMIC_LOADS scenario).

Each live agent's load follows a mean-reverting random walk
(Ornstein-Uhlenbeck, discretised per routed task):

    load <- clip( load + THETA * (base - load) + SIGMA * N(0,1), 0, 1 )

so hot agents stay hot on average but fluctuate. Global information
(GloXRoute's coordinator, DRL-OR-S's link-state table) is only refreshed
every COORDINATOR_UPDATE_INTERVAL tasks, so between refreshes routers act
on slightly stale estimates while the metrics use the true current loads.

Every router is replayed on the SAME load trajectory: reset() restores the
initial loads and re-seeds the noise before each router's pass.
"""
import numpy as np
from core import config as cfg


class LoadDynamics:
    def __init__(self, agents, seed: int, enabled: bool = None):
        self.agents  = agents
        self.seed    = seed
        self.enabled = cfg.DYNAMIC_LOADS if enabled is None else enabled
        self.base    = {n: a.load for n, a in agents.items()}
        self.reset()

    def reset(self):
        for n, a in self.agents.items():
            a.load = self.base[n]
        self.rng = np.random.default_rng(50_000 + self.seed)

    def step(self):
        if not self.enabled:
            return
        noise = self.rng.normal(0.0, 1.0, len(self.agents))
        for (n, a), z in zip(self.agents.items(), noise):
            if getattr(a, "failed", False):
                continue
            a.load = float(np.clip(a.load + cfg.LOAD_THETA * (self.base[n] - a.load) + cfg.LOAD_SIGMA * z, 0.0, 1.0))
