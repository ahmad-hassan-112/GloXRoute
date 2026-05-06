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