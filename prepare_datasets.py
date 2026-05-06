import os
import random
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    SEED,
    NUM_TASKS,
    TRAIN_TASK_RATIO,
    VALIDATION_RATIO,
)
from models import Task
from network_builder import build_ioa_network
from task_generator import generate_tasks
from coordinator import GlobalCoordinator
from nn_learning import NNDatasetBuilder
from gnn_learning import GNNDatasetBuilder


DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")


def tasks_to_dataframe(tasks):
    return pd.DataFrame([asdict(task) for task in tasks])


def dataframe_to_tasks(df):
    return [
        Task(
            task_id=int(row.task_id),
            source=int(row.source),
            required_capability=str(row.required_capability),
            priority=float(row.priority),
            deadline=float(row.deadline),
        )
        for row in df.itertuples(index=False)
    ]


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)

    G, agents = build_ioa_network(seed=SEED)
    all_tasks = generate_tasks(NUM_TASKS)
    random.shuffle(all_tasks)

    total = len(all_tasks)
    train_end = int(TRAIN_TASK_RATIO * total)
    val_end = train_end + int(VALIDATION_RATIO * total)

    train_tasks = all_tasks[:train_end]
    val_tasks = all_tasks[train_end:val_end]
    test_tasks = all_tasks[val_end:]

    tasks_to_dataframe(train_tasks).to_csv(f"{DATASET_DIR}/train_tasks.csv", index=False)
    tasks_to_dataframe(val_tasks).to_csv(f"{DATASET_DIR}/val_tasks.csv", index=False)
    tasks_to_dataframe(test_tasks).to_csv(f"{DATASET_DIR}/test_tasks.csv", index=False)

    coordinator = GlobalCoordinator(G, agents)
    nn_builder = NNDatasetBuilder(G=G, agents=agents, coordinator=coordinator)
    gnn_builder = GNNDatasetBuilder(G=G, agents=agents)

    nn_builder.build(train_tasks, save_path=f"{DATASET_DIR}/nn_train_dataset.csv")
    nn_builder.build(val_tasks, save_path=f"{DATASET_DIR}/nn_val_dataset.csv")
    nn_builder.build(test_tasks, save_path=f"{DATASET_DIR}/nn_test_dataset.csv")

    gnn_builder.build(train_tasks, save_path=f"{DATASET_DIR}/gnn_train_dataset.csv")
    gnn_builder.build(val_tasks, save_path=f"{DATASET_DIR}/gnn_val_dataset.csv")
    gnn_builder.build(test_tasks, save_path=f"{DATASET_DIR}/gnn_test_dataset.csv")

    metadata = {
        "seed": SEED,
        "num_tasks": NUM_TASKS,
        "train_ratio": TRAIN_TASK_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": 1.0 - TRAIN_TASK_RATIO - VALIDATION_RATIO,
        "train_tasks": len(train_tasks),
        "val_tasks": len(val_tasks),
        "test_tasks": len(test_tasks),
    }
    with open(f"{DATASET_DIR}/dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Datasets generated in: {DATASET_DIR}")


if __name__ == "__main__":
    main()
