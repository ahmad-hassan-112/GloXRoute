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
NUM_TASKS  = 2000          # test tasks per seed

TOPOLOGY  = "barabasi_albert"   # "erdos_renyi" | "barabasi_albert" | "watts_strogatz"
BA_M      = 3                   # BA: edges per new node
WS_K      = 6                   # WS: each node connected to k neighbours
WS_BETA   = 0.3                 # WS: rewiring probability

MAX_HOPS  = 20

OUTPUT_DIR = "outputs"

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
LATENCY_BOUND_EPSILON = 0.02
ROUTE_CACHE_MAXSIZE   = 8192