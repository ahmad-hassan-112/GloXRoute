"""
gnn_learning.py — GNN learned scorer (ABLATION ONLY).

Key fixes vs original
---------------------
1. set_graph_context() must be called before train/predict — no silent
   crashes when G/agents are None.
2. coordinator.guidance / .next_hop now use the lazy proxy correctly
   (no .get({}) dict calls on the proxy).
3. Schema version checked on load().
4. NNDatasetBuilder / GNNDatasetBuilder references removed from
   prepare_datasets import chain (was causing circular import).
5. _sample_features() uses the lazy accessor pattern, not dict.get().
"""

import os, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from core.config import (
    CAPABILITIES, PROTOCOLS, OUTPUT_DIR,
    MAX_HOPS, VALIDATION_RATIO,
)

GNN_SCHEMA_VERSION = "v2"


# ================================================================== #
#  Model                                                               #
# ================================================================== #

class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(node_input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, node_features, adj_norm):
        agg1 = torch.matmul(adj_norm, node_features)
        h1   = F.relu(self.fc1(torch.cat([node_features, agg1], dim=1)))
        agg2 = torch.matmul(adj_norm, h1)
        h2   = F.relu(self.fc2(torch.cat([h1, agg2], dim=1)))
        return h2


class GNNCandidateScorer(nn.Module):
    def __init__(self, hidden_dim, capability_dim, sample_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + capability_dim + sample_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_cur, h_cand, cap_onehot, sample_feats):
        x = torch.cat([h_cur, h_cand, cap_onehot, sample_feats], dim=1)
        return self.mlp(x)


# ================================================================== #
#  Learner                                                             #
# ================================================================== #

class GNNHybridLearner:

    SAMPLE_DIM = 4   # lat, bw, global_hint, rec_match

    def __init__(self, hidden_dim=64, epochs=80, batch_size=256, lr=1e-3):
        self.hidden_dim    = hidden_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = lr
        self.schema_version = GNN_SCHEMA_VERSION

        self.cap_to_idx  = {c: i for i, c in enumerate(CAPABILITIES)}
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set by set_graph_context()
        self.G = self.agents = self.coordinator = None
        self.node_features = self.adj_norm = None
        self.edge_lat = self.edge_bw = None
        self.encoder  = self.scorer = None

    # -------------------------------------------------------------- #
    def set_graph_context(self, G, agents, coordinator):
        self.G           = G
        self.agents      = agents
        self.coordinator = coordinator

        self.node_features = self._node_features().to(self.device)
        self.adj_norm      = self._adj_norm().to(self.device)
        self.edge_lat, self.edge_bw = self._edge_tables()
        self.edge_lat = self.edge_lat.to(self.device)
        self.edge_bw  = self.edge_bw.to(self.device)

        node_dim = self.node_features.shape[1]
        cap_dim  = len(CAPABILITIES)
        self.encoder = GraphSAGEEncoder(node_dim, self.hidden_dim).to(self.device)
        self.scorer  = GNNCandidateScorer(
            self.hidden_dim, cap_dim, self.SAMPLE_DIM
        ).to(self.device)

    def _node_features(self):
        n = len(self.G.nodes())
        rows = []
        for node in range(n):
            a = self.agents[node]
            cap_mh  = [1.0 if c in a.capabilities else 0.0 for c in CAPABILITIES]
            prot_oh = [1.0 if a.protocol == p else 0.0 for p in PROTOCOLS]
            rows.append([
                a.reliability, a.resource, a.load,
                self.G.degree[node] / max(1, n),
                *cap_mh, *prot_oh,
            ])
        return torch.tensor(rows, dtype=torch.float32)

    def _adj_norm(self):
        n = len(self.G.nodes())
        adj = np.eye(n, dtype=np.float32)
        for u, v in self.G.edges():
            adj[u, v] = adj[v, u] = 1.0
        adj /= (adj.sum(1, keepdims=True) + 1e-8)
        return torch.tensor(adj, dtype=torch.float32)

    def _edge_tables(self):
        n = len(self.G.nodes())
        lat = np.zeros((n, n), np.float32)
        bw  = np.zeros((n, n), np.float32)
        for u, v in self.G.edges():
            lat[u, v] = lat[v, u] = self.G[u][v]["latency"]  / 10.0
            bw[u, v]  = bw[v, u]  = self.G[u][v]["bandwidth"] / 5.0
        return torch.tensor(lat), torch.tensor(bw)

    # -------------------------------------------------------------- #
    def _prepare(self, df):
        cur  = torch.tensor(df["current_node"].values,   dtype=torch.long)
        cand = torch.tensor(df["candidate_node"].values, dtype=torch.long)
        req  = torch.tensor(
            [self.cap_to_idx[c] for c in df["required_capability"].values],
            dtype=torch.long
        )
        y = torch.tensor(df["label"].values, dtype=torch.float32).view(-1, 1)
        return cur, cand, req, y

    def _sample_features(self, cur_idx, cand_idx, req_idx):
        """Coordinator guidance and recommendation for each sample."""
        lat = self.edge_lat[cur_idx, cand_idx].view(-1, 1)
        bw  = self.edge_bw[cur_idx,  cand_idx].view(-1, 1)

        hints, recs = [], []
        for c, n, r in zip(cur_idx.cpu().tolist(),
                           cand_idx.cpu().tolist(),
                           req_idx.cpu().tolist()):
            cap  = CAPABILITIES[r]
            # Use lazy proxy accessors directly
            hint = self.coordinator.guidance[n][cap]
            rec  = self.coordinator.next_hop[c][cap]
            hints.append(hint)
            recs.append(1.0 if rec == n else 0.0)

        hint_t = torch.tensor(hints, dtype=torch.float32, device=self.device).view(-1, 1)
        rec_t  = torch.tensor(recs,  dtype=torch.float32, device=self.device).view(-1, 1)
        return torch.cat([lat, bw, hint_t, rec_t], dim=1)

    def _forward(self, cur_idx, cand_idx, req_idx):
        assert self.encoder is not None, "Call set_graph_context() first."
        embed    = self.encoder(self.node_features, self.adj_norm)
        h_cur    = embed[cur_idx]
        h_cand   = embed[cand_idx]
        cap_oh   = F.one_hot(req_idx, num_classes=len(CAPABILITIES)).float()
        samp     = self._sample_features(cur_idx, cand_idx, req_idx)
        return self.scorer(h_cur, h_cand, cap_oh, samp)

    # -------------------------------------------------------------- #
    def train(self, train_df, val_df=None, output_dir=OUTPUT_DIR, logger=None):
        assert self.encoder is not None, "Call set_graph_context() first."
        os.makedirs(output_dir, exist_ok=True)
        logger = logger or logging.getLogger(__name__)

        cur_tr, cand_tr, req_tr, y_tr = self._prepare(train_df)
        if val_df is None:
            idx = torch.randperm(len(cur_tr))
            cur_tr,cand_tr,req_tr,y_tr = (t[idx] for t in (cur_tr,cand_tr,req_tr,y_tr))
            n = int(VALIDATION_RATIO * len(cur_tr))
            cur_v,cand_v,req_v,y_v = cur_tr[:n],cand_tr[:n],req_tr[:n],y_tr[:n]
            cur_tr,cand_tr,req_tr,y_tr = cur_tr[n:],cand_tr[n:],req_tr[n:],y_tr[n:]
        else:
            cur_v,cand_v,req_v,y_v = self._prepare(val_df)

        tr_load  = DataLoader(TensorDataset(cur_tr,cand_tr,req_tr,y_tr),
                              batch_size=self.batch_size, shuffle=True)
        val_load = DataLoader(TensorDataset(cur_v,cand_v,req_v,y_v),
                              batch_size=self.batch_size, shuffle=False)

        pos_w = torch.tensor(
            [float((y_tr==0).sum()) / max(1.0, float((y_tr==1).sum()))]
        ).to(self.device)
        crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt   = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.scorer.parameters()),
            lr=self.learning_rate
        )

        rows, best_f1, best_enc, best_scr = [], 0.0, None, None
        for epoch in range(1, self.epochs + 1):
            self.encoder.train(); self.scorer.train()
            losses = []
            for bc, bn, br, by in tr_load:
                bc,bn,br,by = (t.to(self.device) for t in (bc,bn,br,by))
                opt.zero_grad()
                crit(self._forward(bc,bn,br), by).backward()
                opt.step()
                losses.append(crit(self._forward(bc,bn,br), by).item())

            vm = self._eval(val_load, crit)
            if vm["f1"] > best_f1:
                best_f1  = vm["f1"]
                best_enc = {k:v.cpu().clone() for k,v in self.encoder.state_dict().items()}
                best_scr = {k:v.cpu().clone() for k,v in self.scorer.state_dict().items()}
            rows.append({"epoch":epoch,
                         "train_loss":float(np.mean(losses)), **vm})
            logger.info("GNN Epoch %03d/%d | train_loss=%.4f | val_f1=%.4f",
                        epoch, self.epochs, rows[-1]["train_loss"], vm["f1"])

        if best_enc: self.encoder.load_state_dict(best_enc)
        if best_scr: self.scorer.load_state_dict(best_scr)
        pd.DataFrame(rows).to_csv(f"{output_dir}/gnn_epoch_results.csv", index=False)
        self.save(f"{output_dir}/learned_hybrid_xai_gnn.pt")
        return pd.DataFrame(rows)

    def _eval(self, loader, crit):
        self.encoder.eval(); self.scorer.eval()
        losses, preds, trues = [], [], []
        with torch.no_grad():
            for bc,bn,br,by in loader:
                bc,bn,br,by = (t.to(self.device) for t in (bc,bn,br,by))
                logits = self._forward(bc,bn,br)
                losses.append(crit(logits,by).item())
                p = (torch.sigmoid(logits)>=0.5).float()
                preds.extend(p.cpu().numpy().flatten())
                trues.extend(by.cpu().numpy().flatten())
        p,t = np.array(preds), np.array(trues)
        tp=((p==1)&(t==1)).sum(); fp=((p==1)&(t==0)).sum()
        fn=((p==0)&(t==1)).sum(); tn=((p==0)&(t==0)).sum()
        pr=tp/max(1,tp+fp); rc=tp/max(1,tp+fn)
        return {"val_loss":float(np.mean(losses)),
                "val_accuracy":float((tp+tn)/max(1,len(t))),
                "precision":float(pr),"recall":float(rc),
                "f1":float(2*pr*rc/max(1e-8,pr+rc))}

    def predict_score(self, current_node, candidate_node, required_capability):
        assert self.encoder is not None
        self.encoder.eval(); self.scorer.eval()
        with torch.inference_mode():
            c  = torch.tensor([current_node],   dtype=torch.long, device=self.device)
            n  = torch.tensor([candidate_node], dtype=torch.long, device=self.device)
            r  = torch.tensor([self.cap_to_idx[required_capability]],
                              dtype=torch.long, device=self.device)
            return float(torch.sigmoid(self._forward(c,n,r)).item())

    def explain(self, current_node, candidate_node, required_capability):
        hint  = self.coordinator.guidance[candidate_node][required_capability]
        rec   = self.coordinator.next_hop[current_node][required_capability]
        lat   = self.G[current_node][candidate_node]["latency"]
        bw    = self.G[current_node][candidate_node]["bandwidth"]
        return (f"gnn_factors: global_hint={hint:.3f}, "
                f"recommended_match={1.0 if rec==candidate_node else 0.0:.1f}, "
                f"latency={lat:.3f}, bandwidth={bw:.3f}")

    def save(self, path):
        torch.save({
            "encoder_state_dict": self.encoder.state_dict(),
            "scorer_state_dict":  self.scorer.state_dict(),
            "hidden_dim":         self.hidden_dim,
            "schema_version":     self.schema_version,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("schema_version") != self.schema_version:
            raise ValueError(
                f"GNN checkpoint schema '{ckpt.get('schema_version')}' != "
                f"current '{self.schema_version}'. Delete checkpoint and retrain."
            )
        assert self.encoder is not None, "Call set_graph_context() before load()."
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.scorer.load_state_dict(ckpt["scorer_state_dict"])
        self.encoder.to(self.device).eval()
        self.scorer.to(self.device).eval()
        return self


# Alias so old code that imports GNNDatasetBuilder still loads
GNNDatasetBuilder = None   # replaced by prepare_datasets.build_gnn_dataset