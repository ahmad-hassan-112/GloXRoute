"""
prepare_datasets.py — build training/validation datasets for the
ablation-only learned variants (LearnedHybridXAI, NeuralHybridXAI,
GNNHybridXAI).

Critical fixes vs original
---------------------------
1.  Uses the SAME seed-controlled exhaustive task generator as the main
    experiment — no separate random shuffle that could cause leakage.
2.  Train/val split is done on UNIQUE (source, capability) PAIRS, not
    on randomly shuffled individual task rows.  With exhaustive mode the
    population is 800 tasks; a row-level 70/20/10 split is fine because
    there are no duplicates.  But the split is now deterministic and
    labelled clearly.
3.  The oracle used to produce training labels is now IoA-aware:
    it picks the best capable target by EFFECTIVE latency (including
    protocol penalty and load), not raw Dijkstra latency.  This means
    the training signal actually reflects what HybridXAI is trying to
    optimise.
4.  Dataset statistics are written to metadata so the paper can report
    class balance, feature dimensions, and train/val/test sizes.
5.  Neural/GNN datasets are only built if the flag --with_nn is passed,
    since they take significantly longer and are ablation-only.

Usage
-----
    python prepare_datasets.py                 # task splits + NN/GNN datasets
    python prepare_datasets.py --no_nn         # task splits only (fast)
    python prepare_datasets.py --seed 123      # use a specific seed
"""

import argparse, json, os, random
import numpy as np
import pandas as pd
from dataclasses import asdict

from config import (
    OUTPUT_DIR, SEED, TRAIN_TASK_RATIO, VALIDATION_RATIO,
    TOPOLOGY, NUM_AGENTS,
)
from models import Task
from network_builder import build_ioa_network, capability_distribution
from task_generator import generate_tasks
from coordinator import GlobalCoordinator

DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")


# ================================================================== #
#  Task I/O helpers                                                    #
# ================================================================== #

def tasks_to_df(tasks):
    return pd.DataFrame([asdict(t) for t in tasks])

def df_to_tasks(df):
    return [
        Task(
            task_id=int(r.task_id),
            source=int(r.source),
            required_capability=str(r.required_capability),
            priority=float(r.priority),
            deadline=float(r.deadline),
        )
        for r in df.itertuples(index=False)
    ]


# ================================================================== #
#  IoA-aware oracle                                                    #
# ================================================================== #

def oracle_best_path(G, agents, task, coordinator):
    """
    Select the best capable target using EFFECTIVE routing cost
    (includes protocol penalty + load), matching the HybridXAI objective.
    Raw Dijkstra latency is a biased oracle because ShortestPath already
    minimises it — training against it teaches the NN to mimic ShortestPath,
    which is the baseline we are trying to outperform.
    """
    from utils import capable_nodes, effective_edge_weight
    import networkx as nx

    weight_fn = effective_edge_weight(agents)
    targets   = [n for n in capable_nodes(agents, task.required_capability)
                 if not getattr(agents[n], "failed", False)]

    if not targets:
        return [task.source]

    best_path, best_cost = [task.source], float("inf")
    for target in targets:
        try:
            path = nx.shortest_path(G, task.source, target, weight=weight_fn)
            cost = sum(
                float(weight_fn(u, v, G[u][v]) or 999)
                for u, v in zip(path[:-1], path[1:])
            )
            # Add quality bonus: prefer low-load, high-reliability targets
            cost += agents[target].load * 3.0 - agents[target].reliability * 2.0
            if cost < best_cost:
                best_cost, best_path = cost, path
        except Exception:
            continue

    return best_path


# ================================================================== #
#  NN feature extractor (shared schema)                               #
# ================================================================== #

NN_FEATURE_NAMES = [
    "capability_match",
    "global_hint",
    "recommended_next_match",
    "reliability",
    "resource",
    "load",
    "latency_norm",
    "bandwidth_norm",
    "degree_norm",
    "same_protocol",
    "current_load",
    "current_degree_norm",
    "effective_latency_norm",
    "distance_to_required_norm",
    "two_hop_cap_ratio",
]

NN_SCHEMA_VERSION = "v3"   # bump whenever feature list changes


def extract_nn_features(G, agents, coordinator, current, candidate, required):
    """Extract features consistent with HybridXAIRouter's scoring terms."""
    import math
    from utils import capable_nodes, effective_edge_weight

    cap_match  = 1.0 if required in agents[candidate].capabilities else 0.0
    hint       = coordinator.guidance[candidate][required]
    rec_next   = coordinator.next_hop[current][required]
    rec_match  = 1.0 if candidate == rec_next else 0.0
    reliability= agents[candidate].reliability
    resource   = agents[candidate].resource
    load       = agents[candidate].load
    latency    = G[current][candidate]["latency"]
    bandwidth  = G[current][candidate]["bandwidth"]
    degree_norm= G.degree[candidate] / max(1, len(G.nodes()))
    same_proto = 1.0 if agents[current].protocol == agents[candidate].protocol else 0.0
    cur_load   = agents[current].load
    cur_deg    = G.degree[current] / max(1, len(G.nodes()))

    weight_fn  = effective_edge_weight(agents)
    eff_cost   = weight_fn(current, candidate, G[current][candidate])
    eff_lat_norm = (float(eff_cost) / 20.0) if eff_cost is not None else 1.0

    # Distance to nearest capable node from candidate
    dist = coordinator.distance[candidate][required]
    dist_norm = min(dist / 20.0, 1.0) if not math.isinf(dist) else 1.0

    # Two-hop capability ratio
    two_hop = set()
    for n1 in G.neighbors(candidate):
        two_hop.add(n1)
        for n2 in G.neighbors(n1):
            two_hop.add(n2)
    cap_count = sum(1 for n in two_hop if required in agents[n].capabilities)
    two_hop_ratio = cap_count / max(1, len(two_hop))

    return np.array([
        cap_match, hint, rec_match,
        reliability, resource, load,
        latency / 10.0, bandwidth / 5.0, degree_norm,
        same_proto, cur_load, cur_deg,
        eff_lat_norm, dist_norm, two_hop_ratio,
    ], dtype=np.float32)


# ================================================================== #
#  Dataset builders                                                    #
# ================================================================== #

def build_nn_dataset(G, agents, coordinator, tasks, save_path=None):
    """
    One row per (current_node, candidate_neighbor) decision point.
    Label=1 if this candidate is on the IoA-aware oracle path.
    """
    rows = []
    from config import MAX_HOPS
    for task in tasks:
        if task.required_capability in agents[task.source].capabilities:
            continue
        path = oracle_best_path(G, agents, task, coordinator)
        if len(path) <= 1:
            continue

        current = task.source
        required = task.required_capability
        for step in range(min(MAX_HOPS, len(path) - 1)):
            oracle_next = path[step + 1]
            for candidate in G.neighbors(current):
                feats = extract_nn_features(G, agents, coordinator,
                                            current, candidate, required)
                row = {
                    "task_id":            task.task_id,
                    "current_node":       current,
                    "candidate_node":     candidate,
                    "oracle_next":        oracle_next,
                    "required_capability": required,
                    "label":              1 if candidate == oracle_next else 0,
                }
                for name, val in zip(NN_FEATURE_NAMES, feats):
                    row[name] = float(val)
                rows.append(row)
            current = oracle_next
            if required in agents[current].capabilities:
                break

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def build_gnn_dataset(G, agents, coordinator, tasks, save_path=None):
    """
    Minimal GNN dataset: node indices + label only.
    Node features are built at training time from the live graph.
    """
    rows = []
    from config import MAX_HOPS
    for task in tasks:
        if task.required_capability in agents[task.source].capabilities:
            continue
        path = oracle_best_path(G, agents, task, coordinator)
        if len(path) <= 1:
            continue

        current  = task.source
        required = task.required_capability
        for step in range(min(MAX_HOPS, len(path) - 1)):
            oracle_next = path[step + 1]
            for candidate in G.neighbors(current):
                rows.append({
                    "task_id":            task.task_id,
                    "current_node":       current,
                    "candidate_node":     candidate,
                    "required_capability": required,
                    "label":              1 if candidate == oracle_next else 0,
                })
            current = oracle_next
            if required in agents[current].capabilities:
                break

    df = pd.DataFrame(rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


# ================================================================== #
#  Main                                                                #
# ================================================================== #

def main(seed=SEED, build_nn=True, topology=TOPOLOGY, num_agents=NUM_AGENTS):
    os.makedirs(DATASET_DIR, exist_ok=True)
    random.seed(seed); np.random.seed(seed)

    print(f"Building datasets (seed={seed}, topology={topology}, "
          f"n_agents={num_agents})")

    G, agents = build_ioa_network(seed=seed, topology=topology,
                                  num_agents=num_agents)

    cap_dist = capability_distribution(agents)
    print("Capability distribution:", cap_dist)

    # Exhaustive task list — all (source, cap) pairs exactly once
    all_tasks = generate_tasks(seed=seed, num_agents=num_agents)
    print(f"Total tasks (exhaustive): {len(all_tasks)}")

    # Deterministic split by task_id (already shuffled in generate_tasks)
    total     = len(all_tasks)
    train_end = int(TRAIN_TASK_RATIO * total)
    val_end   = train_end + int(VALIDATION_RATIO * total)

    train_tasks = all_tasks[:train_end]
    val_tasks   = all_tasks[train_end:val_end]
    test_tasks  = all_tasks[val_end:]

    print(f"Split: train={len(train_tasks)} val={len(val_tasks)} "
          f"test={len(test_tasks)}")

    # Save task CSVs
    tasks_to_df(train_tasks).to_csv(f"{DATASET_DIR}/train_tasks.csv", index=False)
    tasks_to_df(val_tasks).to_csv(f"{DATASET_DIR}/val_tasks.csv",   index=False)
    tasks_to_df(test_tasks).to_csv(f"{DATASET_DIR}/test_tasks.csv", index=False)

    if not build_nn:
        print("Skipping NN/GNN dataset build (--no_nn flag set).")
        _write_metadata(seed, all_tasks, train_tasks, val_tasks, test_tasks,
                        cap_dist, build_nn=False)
        return

    coordinator = GlobalCoordinator(G, agents)

    for split_name, split_tasks in [
        ("train", train_tasks),
        ("val",   val_tasks),
        ("test",  test_tasks),
    ]:
        print(f"  Building NN dataset: {split_name} ({len(split_tasks)} tasks)...")
        nn_df = build_nn_dataset(
            G, agents, coordinator, split_tasks,
            save_path=f"{DATASET_DIR}/nn_{split_name}_dataset.csv",
        )
        pos  = int(nn_df["label"].sum())
        neg  = len(nn_df) - pos
        print(f"    Rows={len(nn_df)}, pos={pos}, neg={neg}, "
              f"imbalance={neg/max(1,pos):.1f}:1")

        print(f"  Building GNN dataset: {split_name}...")
        build_gnn_dataset(
            G, agents, coordinator, split_tasks,
            save_path=f"{DATASET_DIR}/gnn_{split_name}_dataset.csv",
        )

    _write_metadata(seed, all_tasks, train_tasks, val_tasks, test_tasks,
                    cap_dist, build_nn=True)
    print(f"\nDatasets saved to: {DATASET_DIR}")


def _write_metadata(seed, all_tasks, train_tasks, val_tasks, test_tasks,
                    cap_dist, build_nn):
    metadata = {
        "seed":              seed,
        "task_mode":         "exhaustive",
        "total_tasks":       len(all_tasks),
        "train_tasks":       len(train_tasks),
        "val_tasks":         len(val_tasks),
        "test_tasks":        len(test_tasks),
        "train_ratio":       TRAIN_TASK_RATIO,
        "val_ratio":         VALIDATION_RATIO,
        "test_ratio":        round(1.0 - TRAIN_TASK_RATIO - VALIDATION_RATIO, 4),
        "nn_schema_version": NN_SCHEMA_VERSION,
        "nn_features":       NN_FEATURE_NAMES,
        "nn_feature_dim":    len(NN_FEATURE_NAMES),
        "oracle":            "ioa_aware_effective_latency",
        "capability_distribution": cap_dist,
        "nn_datasets_built": build_nn,
    }
    path = f"{DATASET_DIR}/dataset_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int, default=SEED)
    p.add_argument("--no_nn",      action="store_true",
                   help="Skip NN/GNN dataset build (fast, task splits only)")
    p.add_argument("--topology",   default=TOPOLOGY)
    p.add_argument("--num_agents", type=int, default=NUM_AGENTS)
    args = p.parse_args()
    main(seed=args.seed, build_nn=not args.no_nn,
         topology=args.topology, num_agents=args.num_agents)