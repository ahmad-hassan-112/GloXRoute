import os as _os

SEED = 42

NUM_AGENTS = 100
NUM_TASKS = 20000
MAX_HOPS = 15

OUTPUT_DIR = "outputs"

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

PROTOCOLS = [
    "A2A",
    "MCP",
    "REST",
    "JSON-RPC",
]

COORDINATOR_UPDATE_INTERVAL = 10

# Neural network training settings



NN_HIDDEN_1 = 64
NN_HIDDEN_2 = 32
NN_DROPOUT = 0.10
NN_EPOCHS = 200
NN_BATCH_SIZE = 128
NN_LEARNING_RATE = 0.001

TRAIN_TASK_RATIO = 0.70
VALIDATION_RATIO = 0.20


# improving

# PROTOCOL_MODE = "hard_block"
# FAILURE_RATE = 0.10

# Protocol / heterogeneity settings
PROTOCOL_MODE = "soft_penalty"   # options: "none", "soft_penalty", "hard_block"
PROTOCOL_PENALTY = 2.0

# Congestion settings
CONGESTION_LAMBDA = 1.0

# Evaluation settings
PATH_STRETCH_INF_CAP = 10.0

# Scenario settings
FAILURE_RATE = 0.0
CONGESTION_SCENARIO = False

# GloXRoute-LB settings
LATENCY_BOUND_EPSILON = 0.02
ROUTE_CACHE_MAXSIZE = 8192


def _env_int(name, default):
    return int(_os.environ.get(name, default))


def _env_float(name, default):
    return float(_os.environ.get(name, default))


def _env_bool(name, default):
    value = _os.environ.get(name, None)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y")


SEED = _env_int("GLOX_SEED", SEED)
NUM_AGENTS = _env_int("GLOX_NUM_AGENTS", NUM_AGENTS)
NUM_TASKS = _env_int("GLOX_NUM_TASKS", NUM_TASKS)

PROTOCOL_MODE = _os.environ.get("GLOX_PROTOCOL_MODE", PROTOCOL_MODE)
FAILURE_RATE = _env_float("GLOX_FAILURE_RATE", FAILURE_RATE)
CONGESTION_SCENARIO = _env_bool("GLOX_CONGESTION_SCENARIO", CONGESTION_SCENARIO)

SCENARIO_NAME = _os.environ.get("GLOX_SCENARIO_NAME", "normal")