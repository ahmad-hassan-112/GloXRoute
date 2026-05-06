import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    CAPABILITIES,
    PROTOCOLS,
    OUTPUT_DIR,
    MAX_HOPS,
    VALIDATION_RATIO,
)
from models import Task
from utils import capable_nodes, path_latency


class CentralizedOracle:
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


class GNNDatasetBuilder:
    def __init__(self, G, agents):
        self.G = G
        self.agents = agents
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
                for candidate in self.G.neighbors(current):
                    rows.append(
                        {
                            "task_id": task.task_id,
                            "current_node": current,
                            "candidate_node": candidate,
                            "required_capability": required,
                            "label": 1 if candidate == oracle_next else 0,
                        }
                    )

                current = oracle_next
                if required in self.agents[current].capabilities:
                    break

        df = pd.DataFrame(rows)
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df


class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(node_input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_features, adj_norm):
        agg_1 = torch.matmul(adj_norm, node_features)
        h1 = F.relu(self.fc1(torch.cat([node_features, agg_1], dim=1)))
        agg_2 = torch.matmul(adj_norm, h1)
        h2 = F.relu(self.fc2(torch.cat([h1, agg_2], dim=1)))
        return h2


class GNNCandidateScorer(nn.Module):
    def __init__(self, hidden_dim, capability_dim, sample_feature_dim):
        super().__init__()
        input_dim = (2 * hidden_dim) + capability_dim + sample_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_current, h_candidate, cap_one_hot, sample_features):
        x = torch.cat([h_current, h_candidate, cap_one_hot, sample_features], dim=1)
        return self.mlp(x)


class GNNHybridLearner:
    def __init__(self, hidden_dim=64, epochs=80, batch_size=256, learning_rate=1e-3):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cap_to_idx = {cap: idx for idx, cap in enumerate(CAPABILITIES)}
        self.protocol_to_idx = {p: idx for idx, p in enumerate(PROTOCOLS)}

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.G = None
        self.agents = None
        self.coordinator = None

        self.node_features = None
        self.adj_norm = None
        self.edge_latency = None
        self.edge_bandwidth = None

        self.encoder = None
        self.scorer = None

    def set_graph_context(self, G, agents, coordinator):
        self.G = G
        self.agents = agents
        self.coordinator = coordinator
        self.node_features = self._build_node_features().to(self.device)
        self.adj_norm = self._build_normalized_adjacency().to(self.device)
        self.edge_latency, self.edge_bandwidth = self._build_edge_feature_tables()
        self.edge_latency = self.edge_latency.to(self.device)
        self.edge_bandwidth = self.edge_bandwidth.to(self.device)

        node_input_dim = self.node_features.shape[1]
        capability_dim = len(CAPABILITIES)
        sample_feature_dim = 4
        self.encoder = GraphSAGEEncoder(node_input_dim, self.hidden_dim).to(self.device)
        self.scorer = GNNCandidateScorer(
            self.hidden_dim,
            capability_dim,
            sample_feature_dim
        ).to(self.device)

    def _build_node_features(self):
        n = len(self.G.nodes())
        rows = []
        for node in range(n):
            agent = self.agents[node]
            cap_multi_hot = [
                1.0 if cap in agent.capabilities else 0.0 for cap in CAPABILITIES
            ]
            protocol_one_hot = [
                1.0 if agent.protocol == p else 0.0 for p in PROTOCOLS
            ]
            degree_norm = self.G.degree[node] / max(1, n)
            rows.append(
                [
                    agent.reliability,
                    agent.resource,
                    agent.load,
                    degree_norm,
                    *cap_multi_hot,
                    *protocol_one_hot,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32)

    def _build_normalized_adjacency(self):
        n = len(self.G.nodes())
        adj = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            adj[i, i] = 1.0
            for j in self.G.neighbors(i):
                adj[i, j] = 1.0

        row_sums = adj.sum(axis=1, keepdims=True) + 1e-8
        adj_norm = adj / row_sums
        return torch.tensor(adj_norm, dtype=torch.float32)

    def _build_edge_feature_tables(self):
        n = len(self.G.nodes())
        latency = np.zeros((n, n), dtype=np.float32)
        bandwidth = np.zeros((n, n), dtype=np.float32)

        for u, v in self.G.edges():
            latency_norm = self.G[u][v]["latency"] / 10.0
            bandwidth_norm = self.G[u][v]["bandwidth"] / 5.0
            latency[u, v] = latency_norm
            latency[v, u] = latency_norm
            bandwidth[u, v] = bandwidth_norm
            bandwidth[v, u] = bandwidth_norm

        return (
            torch.tensor(latency, dtype=torch.float32),
            torch.tensor(bandwidth, dtype=torch.float32),
        )

    def _prepare_tensors(self, dataset_df):
        current_idx = torch.tensor(dataset_df["current_node"].values, dtype=torch.long)
        candidate_idx = torch.tensor(dataset_df["candidate_node"].values, dtype=torch.long)
        required_idx = torch.tensor(
            [self.cap_to_idx[c] for c in dataset_df["required_capability"].values],
            dtype=torch.long
        )
        labels = torch.tensor(dataset_df["label"].values, dtype=torch.float32).view(-1, 1)
        return current_idx, candidate_idx, required_idx, labels

    def _sample_features(self, current_idx, candidate_idx, required_idx):
        lat = self.edge_latency[current_idx, candidate_idx].view(-1, 1)
        bw = self.edge_bandwidth[current_idx, candidate_idx].view(-1, 1)

        global_hint = []
        recommended = []
        for c, n, r in zip(
            current_idx.detach().cpu().tolist(),
            candidate_idx.detach().cpu().tolist(),
            required_idx.detach().cpu().tolist(),
        ):
            capability = CAPABILITIES[r]
            hint = self.coordinator.guidance.get(n, {}).get(capability, 0.0)
            rec = self.coordinator.next_hop.get(c, {}).get(capability)
            global_hint.append(hint)
            recommended.append(1.0 if rec == n else 0.0)

        hint_tensor = torch.tensor(global_hint, dtype=torch.float32, device=self.device).view(-1, 1)
        rec_tensor = torch.tensor(recommended, dtype=torch.float32, device=self.device).view(-1, 1)
        return torch.cat([lat, bw, hint_tensor, rec_tensor], dim=1)

    def _forward_batch(self, current_idx, candidate_idx, required_idx):
        node_embed = self.encoder(self.node_features, self.adj_norm)
        h_current = node_embed[current_idx]
        h_candidate = node_embed[candidate_idx]
        cap_one_hot = F.one_hot(required_idx, num_classes=len(CAPABILITIES)).float()
        sample_features = self._sample_features(current_idx, candidate_idx, required_idx)
        return self.scorer(h_current, h_candidate, cap_one_hot, sample_features)

    def train(self, train_df, val_df=None, output_dir=OUTPUT_DIR, logger=None):
        os.makedirs(output_dir, exist_ok=True)
        logger = logger or logging.getLogger(__name__)

        cur_tr, cand_tr, req_tr, y_tr = self._prepare_tensors(train_df)
        if val_df is None:
            indices = torch.randperm(len(cur_tr))
            cur_tr, cand_tr, req_tr, y_tr = cur_tr[indices], cand_tr[indices], req_tr[indices], y_tr[indices]

            val_size = int(VALIDATION_RATIO * len(cur_tr))
            cur_val, cand_val, req_val, y_val = cur_tr[:val_size], cand_tr[:val_size], req_tr[:val_size], y_tr[:val_size]
            cur_tr, cand_tr, req_tr, y_tr = cur_tr[val_size:], cand_tr[val_size:], req_tr[val_size:], y_tr[val_size:]
        else:
            cur_val, cand_val, req_val, y_val = self._prepare_tensors(val_df)

        train_loader = DataLoader(
            TensorDataset(cur_tr, cand_tr, req_tr, y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(cur_val, cand_val, req_val, y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )

        positive_count = max(1.0, float(y_tr.sum().item()))
        negative_count = max(1.0, float(len(y_tr) - y_tr.sum().item()))
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.scorer.parameters()),
            lr=self.learning_rate,
        )

        epoch_rows = []
        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.scorer.train()
            train_losses = []

            for batch_cur, batch_cand, batch_req, batch_y in train_loader:
                batch_cur = batch_cur.to(self.device)
                batch_cand = batch_cand.to(self.device)
                batch_req = batch_req.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self._forward_batch(batch_cur, batch_cand, batch_req)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_metrics = self.evaluate_loader(val_loader, criterion)
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                }
            )

            logger.info(
                f"GNN Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={float(np.mean(train_losses)):.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(f"{output_dir}/gnn_epoch_results.csv", index=False)
        self.save(f"{output_dir}/learned_hybrid_xai_gnn.pt")
        return epoch_df

    def evaluate_loader(self, loader, criterion):
        self.encoder.eval()
        self.scorer.eval()
        losses = []
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_cur, batch_cand, batch_req, batch_y in loader:
                batch_cur = batch_cur.to(self.device)
                batch_cand = batch_cand.to(self.device)
                batch_req = batch_req.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self._forward_batch(batch_cur, batch_cand, batch_req)
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

    def predict_score(self, current_node, candidate_node, required_capability):
        self.encoder.eval()
        self.scorer.eval()
        with torch.no_grad():
            current_idx = torch.tensor([current_node], dtype=torch.long, device=self.device)
            candidate_idx = torch.tensor([candidate_node], dtype=torch.long, device=self.device)
            required_idx = torch.tensor([self.cap_to_idx[required_capability]], dtype=torch.long, device=self.device)
            logit = self._forward_batch(current_idx, candidate_idx, required_idx)
            return float(torch.sigmoid(logit).item())

    def save(self, path):
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "scorer_state_dict": self.scorer.state_dict(),
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.scorer.load_state_dict(checkpoint["scorer_state_dict"])
        self.encoder.to(self.device)
        self.scorer.to(self.device)
        self.encoder.eval()
        self.scorer.eval()
        return self

    def explain(self, current_node, candidate_node, required_capability):
        hint = self.coordinator.guidance.get(candidate_node, {}).get(required_capability, 0.0)
        rec = self.coordinator.next_hop.get(current_node, {}).get(required_capability)
        rec_match = 1.0 if rec == candidate_node else 0.0
        lat = self.G[current_node][candidate_node]["latency"]
        bw = self.G[current_node][candidate_node]["bandwidth"]
        return (
            f"gnn_factors: global_hint={hint:.3f}, recommended_match={rec_match:.1f}, "
            f"latency={lat:.3f}, bandwidth={bw:.3f}"
        )
