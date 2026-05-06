import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    MAX_HOPS,
    OUTPUT_DIR,
    NN_EPOCHS,
    NN_BATCH_SIZE,
    NN_LEARNING_RATE,
    NN_HIDDEN_1,
    NN_HIDDEN_2,
    NN_DROPOUT,
    VALIDATION_RATIO,
)

from models import Task
from utils import capable_nodes, path_latency


class NNFeatureExtractor:
    """
    Extracts candidate next-hop features.
    These are the inputs to the neural network.
    """

    feature_names = [
        "capability_match",
        "global_hint",
        "reliability",
        "resource",
        "load",
        "congestion",
        "latency_norm",
        "bandwidth_norm",
        "degree_norm",

        # New stronger routing features
        "recommended_next_match",
        "distance_to_required_norm",
        "two_hop_capability_count_norm",
        "same_protocol",
        "current_load",
        "current_resource",
        "current_degree_norm",
    ]

    def extract(self, G, agents, coordinator, current_node, candidate_node, required_capability):
        capability_match = 1.0 if required_capability in agents[candidate_node].capabilities else 0.0
        global_hint = coordinator.guidance[candidate_node][required_capability]

        reliability = agents[candidate_node].reliability
        resource = agents[candidate_node].resource
        load = agents[candidate_node].load
        congestion = coordinator.congestion_penalty(candidate_node)

        latency = G[current_node][candidate_node]["latency"]
        bandwidth = G[current_node][candidate_node]["bandwidth"]

        latency_norm = latency / 10.0
        bandwidth_norm = bandwidth / 5.0
        degree_norm = G.degree[candidate_node] / max(1, len(G.nodes()))

         # -------------------------
        # New feature 1:
        # Is this candidate the next hop recommended by global coordinator?
        # -------------------------
        recommended_next_match = 0.0

        if hasattr(coordinator, "next_hop"):
            recommended_next = coordinator.next_hop[current_node][required_capability]
            recommended_next_match = 1.0 if candidate_node == recommended_next else 0.0

        # -------------------------
        # New feature 2:
        # Distance from candidate to nearest capable agent
        # -------------------------
        capable_targets = capable_nodes(agents, required_capability)

        best_distance = float("inf")

        for target in capable_targets:
            try:
                distance = nx.shortest_path_length(
                    G,
                    candidate_node,
                    target,
                    weight="latency"
                )
                best_distance = min(best_distance, distance)
            except nx.NetworkXNoPath:
                continue

        if best_distance == float("inf"):
            distance_to_required_norm = 1.0
        else:
            # Normalize using approximate maximum possible latency path
            distance_to_required_norm = best_distance / (10.0 * max(1, len(G.nodes())))

        # -------------------------
        # New feature 3:
        # How many two-hop neighbors have the required capability?
        # -------------------------
        two_hop_nodes = set()

        for n1 in G.neighbors(candidate_node):
            two_hop_nodes.add(n1)
            for n2 in G.neighbors(n1):
                two_hop_nodes.add(n2)

        two_hop_capability_count = sum(
            1 for node in two_hop_nodes
            if required_capability in agents[node].capabilities
        )

        two_hop_capability_count_norm = two_hop_capability_count / max(1, len(two_hop_nodes))

        # -------------------------
        # New feature 4:
        # Protocol compatibility
        # -------------------------
        same_protocol = 1.0 if agents[current_node].protocol == agents[candidate_node].protocol else 0.0

        # -------------------------
        # New feature 5:
        # Current agent condition
        # -------------------------
        current_load = agents[current_node].load
        current_resource = agents[current_node].resource
        current_degree_norm = G.degree[current_node] / max(1, len(G.nodes()))

        return np.array([
            capability_match,
            global_hint,
            reliability,
            resource,
            load,
            congestion,
            latency_norm,
            bandwidth_norm,
            degree_norm,
            recommended_next_match,
            distance_to_required_norm,
            two_hop_capability_count_norm,
            same_protocol,
            current_load,
            current_resource,
            current_degree_norm,
        ], dtype=np.float32)


class CentralizedOracle:
    """
    Teacher/oracle.
    It has full network information.
    We use it only to generate training labels.
    """

    def best_path(self, G, agents, task: Task):
        source = task.source
        required = task.required_capability

        candidates = capable_nodes(agents, required)

        best_path = [source]
        best_score = float("inf")

        for target in candidates:
            try:
                path = nx.shortest_path(G, source, target, weight="latency")

                latency = path_latency(G, agents, path)
                load_penalty = agents[target].load * 5.0
                reliability_bonus = agents[target].reliability * 3.0
                resource_bonus = agents[target].resource * 2.0

                score = latency + load_penalty - reliability_bonus - resource_bonus

                if score < best_score:
                    best_score = score
                    best_path = path

            except nx.NetworkXNoPath:
                continue

        return best_path


class NNDatasetBuilder:
    """
    Builds the supervised dataset.

    Each row is one candidate neighbor.
    Label = 1 if centralized oracle selected this candidate as next hop.
    Label = 0 otherwise.
    """

    def __init__(self, G, agents, coordinator):
        self.G = G
        self.agents = agents
        self.coordinator = coordinator
        self.extractor = NNFeatureExtractor()
        self.oracle = CentralizedOracle()

    def build(self, tasks, save_path=None):
        rows = []

        for task in tasks:
            oracle_path = self.oracle.best_path(self.G, self.agents, task)

            if len(oracle_path) <= 1:
                continue

            current = task.source
            required = task.required_capability

            for step in range(min(MAX_HOPS, len(oracle_path) - 1)):
                oracle_next = oracle_path[step + 1]
                neighbors = list(self.G.neighbors(current))

                for candidate in neighbors:
                    features = self.extractor.extract(
                        self.G,
                        self.agents,
                        self.coordinator,
                        current,
                        candidate,
                        required,
                    )

                    label = 1 if candidate == oracle_next else 0

                    row = {
                        "task_id": task.task_id,
                        "current_node": current,
                        "candidate_node": candidate,
                        "oracle_next": oracle_next,
                        "required_capability": required,
                        "label": label,
                    }

                    for name, value in zip(NNFeatureExtractor.feature_names, features):
                        row[name] = float(value)

                    rows.append(row)

                current = oracle_next

                if required in self.agents[current].capabilities:
                    break

        df = pd.DataFrame(rows)

        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df


class CandidateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class NextHopMLP(nn.Module):
    """
    Neural network for next-hop selection.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, NN_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(NN_DROPOUT),

            nn.Linear(NN_HIDDEN_1, NN_HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(NN_DROPOUT),

            nn.Linear(NN_HIDDEN_2, 1)
        )

    def forward(self, x):
        return self.net(x)


class NeuralHybridLearner:
    """
    Neural network learner for Hybrid-XAI routing.
    """

    def __init__(self):
        self.feature_names = NNFeatureExtractor.feature_names
        self.input_dim = len(self.feature_names)

        self.model = NextHopMLP(self.input_dim)

        self.mean = None
        self.std = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def prepare_data(self, train_df, val_df=None):
        X_train = train_df[self.feature_names].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.float32)

        if val_df is None:
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            val_size = int(VALIDATION_RATIO * len(X_train))
            X_val = X_train[:val_size]
            y_val = y_train[:val_size]
            X_train = X_train[val_size:]
            y_train = y_train[val_size:]
        else:
            X_val = val_df[self.feature_names].values.astype(np.float32)
            y_val = val_df["label"].values.astype(np.float32)

        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0) + 1e-8

        X_train = (X_train - self.mean) / self.std
        X_val = (X_val - self.mean) / self.std

        return X_train, y_train, X_val, y_val

    def train(self, train_df, val_df=None, output_dir=OUTPUT_DIR, logger=None):
        os.makedirs(output_dir, exist_ok=True)
        logger = logger or logging.getLogger(__name__)

        X_train, y_train, X_val, y_val = self.prepare_data(train_df, val_df)

        train_dataset = CandidateDataset(X_train, y_train)
        val_dataset = CandidateDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=NN_BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=NN_BATCH_SIZE,
            shuffle=False
        )

        positive_count = max(1.0, float(y_train.sum()))
        negative_count = max(1.0, float(len(y_train) - y_train.sum()))
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=NN_LEARNING_RATE)

        epoch_rows = []

        for epoch in range(1, NN_EPOCHS + 1):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))

            val_metrics = self.evaluate_loader(val_loader, criterion)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }

            epoch_rows.append(row)

            logger.info(
                f"Epoch {epoch:03d}/{NN_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(f"{output_dir}/nn_epoch_results.csv", index=False)

        self.save(f"{output_dir}/learned_hybrid_xai_nn.pt")

        return epoch_df

    def evaluate_loader(self, loader, criterion):
        self.model.eval()

        losses = []
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy().flatten().tolist())
                all_true.extend(batch_y.cpu().numpy().flatten().tolist())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        tp = np.sum((all_preds == 1) & (all_true == 1))
        tn = np.sum((all_preds == 0) & (all_true == 0))
        fp = np.sum((all_preds == 1) & (all_true == 0))
        fn = np.sum((all_preds == 0) & (all_true == 1))

        accuracy = (tp + tn) / max(1, len(all_true))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        return {
            "loss": float(np.mean(losses)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def normalize_features(self, features):
        features = np.array(features, dtype=np.float32)
        return (features - self.mean) / self.std

    def predict_score(self, features):
        self.model.eval()

        features = self.normalize_features(features)
        x = torch.tensor(features, dtype=torch.float32).view(1, -1).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()

        return prob

    def explain(self, features):
        """
        Gradient-based XAI explanation.
        It estimates which features most influenced the NN decision.
        """

        self.model.eval()

        raw_features = np.array(features, dtype=np.float32)
        norm_features = self.normalize_features(raw_features)

        x = torch.tensor(norm_features, dtype=torch.float32).view(1, -1).to(self.device)
        x.requires_grad = True

        logit = self.model(x)
        logit.backward()

        gradients = x.grad.detach().cpu().numpy().flatten()
        contributions = gradients * norm_features

        ranked = sorted(
            zip(self.feature_names, contributions),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        top_parts = [
            f"{name}={value:.3f}"
            for name, value in ranked[:4]
        ]

        return "top_nn_factors: " + ", ".join(top_parts)

    def save(self, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "feature_names": self.feature_names,
            "input_dim": self.input_dim,
        }

        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.input_dim = checkpoint["input_dim"]
        self.feature_names = checkpoint["feature_names"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]

        self.model = NextHopMLP(self.input_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        return self