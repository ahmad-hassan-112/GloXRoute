# ============================================================
# GloXRoute — configuration
# ============================================================
import random, numpy as np

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42          # default single seed; multi-seed runner overrides this
SEEDS = [42, 123, 456, 789, 1234, 2024, 3141, 9999, 31415, 27182,
         11, 22, 33, 44, 55, 66, 77, 88, 99, 100]   # 20 seeds

# ------------------------------------------------------------------
# Network  (will be overridden by topology experiments)
# ------------------------------------------------------------------
NUM_AGENTS = 100

# Task generation mode:
#   "exhaustive" — enumerate ALL (source, capability) pairs. Deterministic,
#                  zero within-seed sampling noise, all variance comes from
#                  seeds where it belongs. RECOMMENDED for the paper.
#   "random"     — legacy random sampling (kept for backwards compatibility)
TASK_MODE = "exhaustive"
NUM_TASKS = 2000           # only used when TASK_MODE == "random"

TOPOLOGY  = "barabasi_albert"   # "erdos_renyi" | "barabasi_albert" | "watts_strogatz"
BA_M      = 3                   # BA: edges per new node
WS_K      = 6                   # WS: each node connected to k neighbours
WS_BETA   = 0.3                 # WS: rewiring probability

MAX_HOPS  = 20

import os as _os
# All output paths are relative to OUTPUT_DIR. This can be overridden by
# setting the GLOXROUTE_RUN_DIR environment variable, which the Makefile
# sets to a timestamped folder (outputs/runs/<YYYY-MM-DD_HH-MM-SS>) so
# every `make all_modes` run is kept separately and never overwrites
# a previous run's results.
import time as _time
# Direct runs (python3 main.py ...) also get a timestamped folder.
OUTPUT_DIR = _os.environ.get(
    "GLOXROUTE_RUN_DIR",
    _os.path.join("outputs", _time.strftime("%Y-%m-%d_%H-%M-%S")),
)

# ------------------------------------------------------------------
# Capabilities  — kept sparse so routing is non-trivial
# ------------------------------------------------------------------
CAPABILITIES = [
    "search",
    "reasoning",
    "coding",
    "math",
    "planning",
    "translation",
    "vision",
    "storage",
]

# How many capabilities each agent gets (1 = scarce, harder routing)
CAPS_PER_AGENT_MIN = 1
CAPS_PER_AGENT_MAX = 1    # keep at 1 for hard/scarce regime

# Capability SCARCITY (applied to AGENTS, not to task requests).
# Agents are assigned capabilities following a Zipf distribution so that
# some capabilities are held by very few agents. This is what makes
# multi-hop routing genuinely necessary.
#   exponent 0.0 -> uniform (~12 agents/cap, trivially easy)
#   exponent 1.5 -> 52,18,10,7,5,4,3,2 agents/cap (hard, realistic)
CAPABILITY_ZIPF_EXPONENT   = 1.5
MIN_AGENTS_PER_CAPABILITY  = 2    # guarantee every capability is reachable

PROTOCOLS = ["A2A", "MCP", "REST", "JSON-RPC"]

# ------------------------------------------------------------------
# Scenario knobs
# ------------------------------------------------------------------
FAILURE_RATE        = 0.0      # fraction of agents that fail
CONGESTION_SCENARIO = False    # if True, 30 % of agents get load drawn from Beta(5,1)
CONGESTION_FRACTION = 0.30     # fraction of agents that are "hot" under congestion
COORDINATOR_UPDATE_INTERVAL = 10

# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------
PROTOCOL_MODE    = "soft_penalty"   # "none" | "soft_penalty" | "hard_block"
PROTOCOL_PENALTY = 2.0
CONGESTION_LAMBDA = 1.0

# ------------------------------------------------------------------
# Neural / GNN  (kept for legacy; not used in main experiments)
# ------------------------------------------------------------------
NN_HIDDEN_1      = 64
NN_HIDDEN_2      = 32
NN_DROPOUT       = 0.10
NN_EPOCHS        = 200
NN_BATCH_SIZE    = 128
NN_LEARNING_RATE = 0.001

TRAIN_TASK_RATIO  = 0.70
VALIDATION_RATIO  = 0.20

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
PATH_STRETCH_INF_CAP = 10.0

# Tie-break window for the glass-box scoring function: candidates within
# (1 + LATENCY_BOUND_EPSILON) x the best path-cost are all considered
# "near-optimal" and compete on the interpretable score terms (capability,
# reliability, load, protocol, global hint).
#
# At 0.02 (2%) almost every hop has only ONE feasible candidate, so the
# score terms never actually get to break a tie -- this was silently
# making the ablation study meaningless (removing load/protocol/reliability
# terms showed ~0% effect because they were never being evaluated).
# Widened to 0.15 so multiple candidates are typically in contention,
# which is also more representative of real IoA routing where several
# next hops are "good enough" on latency and the interesting decision is
# which one is best on capability/state/protocol grounds.
LATENCY_BOUND_EPSILON = 0.15

# On low-degree graphs (mean degree ~6), a percentage window alone often
# admits only 1 candidate even at 15%, because absolute cost gaps between
# neighbours can be large. TIE_BREAK_TOP_K guarantees at least this many
# candidates (or all available, if fewer) always enter the score-based
# tie-break, so the interpretable score terms are exercised even when the
# percentage window is too narrow. This does NOT change which candidates
# are "feasible" for latency purposes -- it only guarantees the ablation
# study and scoring function actually get evaluated on >1 option per hop.
TIE_BREAK_TOP_K = 3
ROUTE_CACHE_MAXSIZE   = 8192

# ------------------------------------------------------------------
# Recent CCF-A learned baselines (core/learned_routers.py)
# ------------------------------------------------------------------
# DRL-OR-S (TPDS 2023).  "transfer": train once per configuration on
# DRLOR_TRAIN_GRAPHS graphs whose seeds are disjoint from SEEDS, then
# evaluate the frozen policy on every evaluation seed (no test leakage).
# "same_graph": train on each evaluation graph with a separate task stream.
DRLOR_TRAIN_MODE        = "transfer"
DRLOR_TRAIN_EPISODES    = 6000
DRLOR_TRAIN_GRAPHS      = 5
DRLOR_TRAIN_SEED_OFFSET = 900_000
DRLOR_HIDDEN            = 32
DRLOR_LR                = 3e-3
DRLOR_ENTROPY           = 0.01
DRLOR_BATCH             = 16

# AgentNet (NeurIPS 2025).  Warm-up tasks served on the evaluation graph
# before measurement (random stream, separate RNG); learning continues online.
AGENTNET_WARMUP_TASKS    = 3000
AGENTNET_MEMORY_SIZE     = 64      # records per agent
AGENTNET_PRUNE_THRESHOLD = 0.25    # link weight below which a link is pruned

# ------------------------------------------------------------------
# Dynamic load scenario (core/dynamics.py) — enabled with main.py --dynamic
# ------------------------------------------------------------------
DYNAMIC_LOADS = False
LOAD_THETA    = 0.10     # mean reversion per task
LOAD_SIGMA    = 0.08     # per-task noise (stationary SD ~ 0.18)

# ------------------------------------------------------------------
# Stochastic delivery delay (always on; core/metrics.py)
# ------------------------------------------------------------------
# Each hop's realised delay = expected effective cost x a mean-1 log-normal
# factor with this coefficient of variation. Routers plan with expected
# values; deadline_miss is judged on the realised delay. Factors are drawn
# from common random numbers keyed on (seed, task, hop), so every method
# faces identical jitter on identical hops. 0.0 disables jitter.
DELAY_JITTER_CV = 0.20
