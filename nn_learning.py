"""
nn_learning.py — Neural network learned scorer (ABLATION ONLY).

This module is NOT part of the main GloXRoute comparison.
It is an ablation variant that replaces the glass-box linear scoring
function with a small MLP, at the cost of interpretability.

Key fixes vs original
---------------------
1. Feature schema now matches prepare_datasets.NN_FEATURE_NAMES exactly.
   The original had an 18-feature extractor that silently diverged from
   the training schema, causing the runtime ValueError: shapes (18,)≠(20,).
2. Schema version is checked on load() so stale checkpoints are rejected
   with a clear error rather than silent corruption.
3. Training uses the IoA-aware oracle labels from prepare_datasets,
   not the raw-latency oracle.
4. Class-weighted loss and early stopping on val_f1 are both kept.
5. explain() uses gradient×input (saliency) which is correct for MLP.
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    OUTPUT_DIR, NN_EPOCHS, NN_BATCH_SIZE, NN_LEARNING_RATE,
    NN_HIDDEN_1, NN_HIDDEN_2, NN_DROPOUT, VALIDATION_RATIO,
)

# Import the canonical feature list from prepare_datasets
try:
    from prepare_datasets import NN_FEATURE_NAMES, NN_SCHEMA_VERSION, extract_nn_features
except ImportError:
    # fallback for standalone use
    NN_FEATURE_NAMES = [
        "capability_match", "global_hint", "recommended_next_match",
        "reliability", "resource", "load",
        "latency_norm", "bandwidth_norm", "degree_norm",
        "same_protocol", "current_load", "current_degree_norm",
        "effective_latency_norm", "distance_to_required_norm", "two_hop_cap_ratio",
    ]
    NN_SCHEMA_VERSION = "v3"
    from utils import capable_nodes, effective_edge_weight
    import math

    def extract_nn_features(G, agents, coordinator, current, candidate, required):
        import math
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
        wf = effective_edge_weight(agents)
        ec = wf(current, candidate, G[current][candidate])
        eff_lat = (float(ec)/20.0) if ec is not None else 1.0
        dist = coordinator.distance[candidate][required]
        dist_norm = min(dist/20.0,1.0) if not math.isinf(dist) else 1.0
        two_hop = set()
        for n1 in G.neighbors(candidate):
            two_hop.add(n1)
            for n2 in G.neighbors(n1): two_hop.add(n2)
        ratio = sum(1 for n in two_hop if required in agents[n].capabilities)/max(1,len(two_hop))
        return np.array([cap_match,hint,rec_match,reliability,resource,load,
                         latency/10,bandwidth/5,degree_norm,same_proto,
                         cur_load,cur_deg,eff_lat,dist_norm,ratio],dtype=np.float32)


# Re-export for routers.py
NNFeatureExtractor_extract = extract_nn_features


class NNFeatureExtractor:
    """Thin wrapper so routers.py interface stays unchanged."""
    feature_names = NN_FEATURE_NAMES

    def extract(self, G, agents, coordinator, current_node, candidate_node,
                required_capability):
        return extract_nn_features(G, agents, coordinator,
                                   current_node, candidate_node,
                                   required_capability)


# ================================================================== #
#  Dataset                                                             #
# ================================================================== #

class CandidateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ================================================================== #
#  Model                                                               #
# ================================================================== #

class NextHopMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, NN_HIDDEN_1), nn.ReLU(), nn.Dropout(NN_DROPOUT),
            nn.Linear(NN_HIDDEN_1, NN_HIDDEN_2), nn.ReLU(), nn.Dropout(NN_DROPOUT),
            nn.Linear(NN_HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================== #
#  Learner                                                             #
# ================================================================== #

class NeuralHybridLearner:

    def __init__(self):
        self.feature_names   = NN_FEATURE_NAMES
        self.schema_version  = NN_SCHEMA_VERSION
        self.input_dim       = len(self.feature_names)
        self.model           = NextHopMLP(self.input_dim)
        self.mean: np.ndarray = None
        self.std:  np.ndarray = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # -------------------------------------------------------------- #
    def _split(self, df, val_df):
        X = df[self.feature_names].values.astype(np.float32)
        y = df["label"].values.astype(np.float32)
        if val_df is not None:
            Xv = val_df[self.feature_names].values.astype(np.float32)
            yv = val_df["label"].values.astype(np.float32)
        else:
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
            n = int(VALIDATION_RATIO * len(X))
            Xv, yv = X[:n], y[:n]
            X,  y  = X[n:], y[n:]
        self.mean = X.mean(0); self.std = X.std(0) + 1e-8
        return (X - self.mean)/self.std, y, (Xv - self.mean)/self.std, yv

    def train(self, train_df, val_df=None, output_dir=OUTPUT_DIR, logger=None):
        os.makedirs(output_dir, exist_ok=True)
        logger = logger or logging.getLogger(__name__)

        # Validate schema
        missing = set(self.feature_names) - set(train_df.columns)
        if missing:
            raise ValueError(f"Training data missing features: {missing}. "
                             "Re-run prepare_datasets.py.")

        X_tr, y_tr, X_val, y_val = self._split(train_df, val_df)

        tr_loader  = DataLoader(CandidateDataset(X_tr, y_tr),
                                batch_size=NN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(CandidateDataset(X_val, y_val),
                                batch_size=NN_BATCH_SIZE, shuffle=False)

        pos_w = torch.tensor(
            [float((y_tr == 0).sum()) / max(1.0, float((y_tr == 1).sum()))],
            dtype=torch.float32
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=NN_LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10
        )

        best_f1, best_state, rows = 0.0, None, []
        for epoch in range(1, NN_EPOCHS + 1):
            self.model.train()
            losses = []
            for bx, by in tr_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward(); optimizer.step()
                losses.append(loss.item())

            vm = self._eval(val_loader, criterion)
            scheduler.step(vm["f1"])
            if vm["f1"] > best_f1:
                best_f1 = vm["f1"]
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}

            rows.append({"epoch": epoch,
                         "train_loss": float(np.mean(losses)), **vm})
            logger.info("Epoch %03d/%d | train_loss=%.4f | val_f1=%.4f",
                        epoch, NN_EPOCHS, rows[-1]["train_loss"], vm["f1"])

        if best_state:
            self.model.load_state_dict(best_state)
        pd.DataFrame(rows).to_csv(f"{output_dir}/nn_epoch_results.csv", index=False)
        self.save(f"{output_dir}/learned_hybrid_xai_nn.pt")
        return pd.DataFrame(rows)

    def _eval(self, loader, criterion):
        self.model.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = self.model(bx)
                losses.append(criterion(logits, by).item())
                p = (torch.sigmoid(logits) >= 0.5).float()
                preds.extend(p.cpu().numpy().flatten())
                trues.extend(by.cpu().numpy().flatten())
        p, t = np.array(preds), np.array(trues)
        tp = ((p==1)&(t==1)).sum(); fp = ((p==1)&(t==0)).sum()
        fn = ((p==0)&(t==1)).sum(); tn = ((p==0)&(t==0)).sum()
        prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
        return {
            "val_loss":     float(np.mean(losses)),
            "val_accuracy": float((tp+tn)/max(1,len(t))),
            "precision":    float(prec),
            "recall":       float(rec),
            "f1":           float(2*prec*rec/max(1e-8,prec+rec)),
        }

    def normalize(self, features):
        return (np.array(features, dtype=np.float32) - self.mean) / self.std

    def predict_score(self, features):
        self.model.eval()
        x = torch.tensor(self.normalize(features)).view(1, -1).to(self.device)
        with torch.inference_mode():
            return float(torch.sigmoid(self.model(x)).item())

    def explain(self, features):
        """Gradient × input saliency for the MLP."""
        self.model.eval()
        norm = self.normalize(features)
        x = torch.tensor(norm, dtype=torch.float32).view(1, -1).to(self.device)
        x.requires_grad_(True)
        self.model(x).backward()
        contrib = (x.grad.detach().cpu().numpy().flatten() * norm)
        ranked = sorted(zip(self.feature_names, contrib),
                        key=lambda kv: abs(kv[1]), reverse=True)
        return "top_nn_factors: " + ", ".join(
            f"{n}={v:.3f}" for n, v in ranked[:4]
        )

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mean":             self.mean,
            "std":              self.std,
            "feature_names":    self.feature_names,
            "schema_version":   self.schema_version,
            "input_dim":        self.input_dim,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("schema_version") != self.schema_version:
            raise ValueError(
                f"NN checkpoint schema '{ckpt.get('schema_version')}' != "
                f"current '{self.schema_version}'. Delete checkpoint and retrain."
            )
        self.feature_names = ckpt["feature_names"]
        self.input_dim     = ckpt["input_dim"]
        self.mean          = ckpt["mean"]
        self.std           = ckpt["std"]
        self.model         = NextHopMLP(self.input_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        return self


# Legacy alias kept for backwards compat with old experiment.py
NNDatasetBuilder = None   # replaced by prepare_datasets.build_nn_dataset