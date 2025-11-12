# %%
"""
GraphSAGE on a kNN graph for rare-event node classification
-----------------------------------------------------------
- Builds a kNN graph from your feature matrix X (R^{N x d})
- Trains a GraphSAGE GNN with BCEWithLogitsLoss and pos_weight for imbalance
- Reports ROC-AUC and PR-AUC (Average Precision) and saves diagnostic plots

Usage
-----
1) Set `use_files = True` in `main()` to read your files with `_load_data_to_mem`.
2) Fill `parameter_config` and `files` accordingly.
3) Run:  python knn-graphsage-node-classification.py

Dependencies
------------
- torch, torch_geometric (PyG)
- numpy, pandas, h5py
- scikit-learn
- matplotlib (for plots)
"""
from __future__ import annotations
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Union
from resolve.utilities import utilities as utils
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import h5py
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.utils import shuffle
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import add_self_loops, to_undirected
from torch_geometric.utils import dropout_edge
import matplotlib.pyplot as plt

# %%
import hnswlib, numpy as np, torch
from torch_geometric.utils import to_undirected, add_self_loops



# %%
def focal_bce_with_logits(logits, targets, alpha=0.5, gamma=2.0):
    p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    loss_pos = -alpha * ((1-p)**gamma) * targets * torch.log(p)
    loss_neg = -(1-alpha) * (p**gamma) * (1-targets) * torch.log(1-p)
    return (loss_pos + loss_neg).mean()

# %%



# ------------------------------
# Repro
# ------------------------------
from torch import logit


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_hdf5_files(path_to_files, config_file):
        return sorted(str(p) for p in path_to_files.glob(f"*.{config_file['simulation_settings']['file_format']}"))

# ------------------------------
# Data utilities + file pipeline
# ------------------------------
def _read_in_from_file(file_path: str, parameter_config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read (theta, phi, y) from HDF5 or CSV and return tensors (x, y).
    - x = [theta | phi] concatenated along last dimension, float32
    - y shaped (N, 1) float32

    parameter_config can provide either 'key'+'selected_indices' (for HDF5) or 'selected_labels' (for CSV):
      'phi':   {'key' or 'selected_labels', 'selected_indices'}
      'theta': {'key' or 'selected_labels', 'selected_indices'}
      'target':{'key' or 'selected_labels', 'selected_indices'}
    """
    if file_path.endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as hdf:
            # φ
            phi = hdf[parameter_config['phi']['key']][:, parameter_config['phi']['selected_indices']]
            # θ
            theta = hdf[parameter_config['theta']['key']]
            if len(parameter_config['theta']['selected_indices']) != 0:
                if theta.ndim == 1:
                    theta_vec = theta[parameter_config['theta']['selected_indices']]  # (T,)
                    theta = torch.from_numpy(theta_vec).unsqueeze(0).expand(phi.shape[0], -1)
                else:
                    theta = theta[:, parameter_config['theta']['selected_indices']]
                    theta = torch.from_numpy(theta)
            else:
                theta = torch.from_numpy(theta)

            # y / target
            tgt_ds = hdf[parameter_config['target']['key']]
            if tgt_ds.ndim > 1 and parameter_config['target']['selected_indices'] is not None:
                y = tgt_ds[:, parameter_config['target']['selected_indices']]
            else:
                y = tgt_ds[:].reshape(-1, 1)

        phi = torch.from_numpy(phi)
        y = torch.from_numpy(y)
        x = torch.cat([theta, phi], dim=-1)

    elif file_path.endswith('.csv'):
        # CSV via selected_labels
        df = pd.read_csv(file_path)

        def select_labels(df_: pd.DataFrame, labels: Union[str, Sequence[str]]) -> pd.DataFrame:
            if isinstance(labels, str):
                return df_[[labels]]
            elif isinstance(labels, (list, tuple)):
                return df_[list(labels)]
            else:
                raise ValueError(f"Invalid label type: {type(labels)}")

        phi_df = select_labels(df, parameter_config['phi']['selected_labels'])
        theta_df = select_labels(df, parameter_config['theta']['selected_labels'])
        y_df = select_labels(df, parameter_config['target']['selected_labels'])

        phi = torch.tensor(phi_df.values, dtype=torch.float32)
        theta = torch.tensor(theta_df.values, dtype=torch.float32)
        y = torch.tensor(y_df.values, dtype=torch.float32)

        if y.ndim == 1:
            y = y.unsqueeze(1)

        x = torch.cat([theta, phi], dim=-1)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # ensure float32 tensors
    x = x.contiguous().to(torch.float32)
    y = y.contiguous().to(torch.float32)
    return x, y


def _load_data_to_mem(files: Sequence[str], cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, ys, file_inds = [], [], []
    for i, fp in enumerate(files):
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        Xi, yi = _read_in_from_file(fp, cfg)
        X.append(Xi)
        ys.append(yi)
        file_inds.append(torch.full((Xi.size(0),), i, dtype=torch.long))
    x = torch.cat(X, 0).contiguous()
    y = torch.cat(ys, 0).contiguous()
    fidx = torch.cat(file_inds, 0).contiguous()
    return x, y, fidx





    




def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# %%
def read_in_data():
        # Example parameter_config (edit to your keys/labels/indices)
        path_to_settings = "./binary-black-hole"
        with open(f"{path_to_settings}/settings.yaml", "r") as f:
            config_file = yaml.safe_load(f)
        sim = config_file["simulation_settings"]
        parameters = {
            "phi":    {"key": "phi",    "label_key": "phi_labels",    "selected_labels": sim["phi_labels"],    "size": len(sim["phi_labels"]),      "selected_indices": None},
            "theta":  {"key": "theta",  "label_key": "theta_headers", "selected_labels": sim["theta_labels"],  "size": len(sim["theta_labels"]),  "selected_indices": None},
            "target": {"key": "target", "label_key": "target_headers","selected_labels": sim["target_labels"], "size": len(sim["target_labels"]), "selected_indices": None},
        }
        files = _get_hdf5_files(Path(config_file["path_settings"]["path_to_files_train"]), config_file)

        if files[0].endswith(('.h5', '.hdf5')):
            parameters["phi"]["selected_indices"] = utils.find_selected_indices(files[0],parameters["phi"])
            parameters["target"]["selected_indices"] = utils.find_selected_indices(files[0],parameters["target"])
            parameters["theta"]["selected_indices"] = utils.find_selected_indices(files[0],parameters["theta"])


        X_t, y_t, _ = _load_data_to_mem(files, parameters)
        # Ensure numpy arrays for kNN builder

        # X is your raw numpy or torch input, shape [N, D]
        scaler = StandardScaler()

        # Convert to numpy if needed
        X_np = X_t.cpu().numpy() if isinstance(X_t, torch.Tensor) else X_t
        #X_t = X.detach().clone().to(torch.float32) if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)

        # Fit on ALL data (or train split only)
        X_np = scaler.fit_transform(X_np)

        # Back to torch
        X = X_np
        y_arr = y_t.cpu().numpy().reshape(-1, 1)

        # Ensure binary labels {0,1}. If multi-target or continuous, map/threshold here.
        if y_arr.shape[1] > 1:
            # choose a column or reduce to a binary indicator
            y = (y_arr[:, 0] > 0.5).astype(np.int64)
        else:
            if not np.array_equal(np.unique(y_arr), np.array([0, 1])):
                y = (y_arr[:, 0] > 0.5).astype(np.int64)
            else:
                y = y_arr[:, 0].astype(np.int64)
        
        X, y = shuffle(X, y, random_state=42)
        return X[:50000,:], y[:50000]

# %%


# %%



def plot_pr_curve(y_true_np, probs_np, outdir="plots", title_prefix="Test"):
    _ensure_dir(outdir)
    precision, recall, _ = precision_recall_curve(y_true_np, probs_np)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} PR Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_pr_curve.png"), dpi=150)
    plt.close()


def plot_roc_curve(y_true_np, probs_np, outdir="plots", title_prefix="Test"):
    _ensure_dir(outdir)
    fpr, tpr, _ = roc_curve(y_true_np, probs_np)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title_prefix} ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_roc_curve.png"), dpi=150)
    plt.close()


def plot_score_hist(probs_np, y_true_np, outdir="plots", title_prefix="Test"):
    _ensure_dir(outdir)
    plt.figure()
    plt.hist(probs_np[y_true_np == 0], bins=50, alpha=0.7, label="neg")
    plt.hist(probs_np[y_true_np == 1], bins=50, alpha=0.7, label="pos")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.yscale('log')
    plt.title(f"{title_prefix} Score Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title_prefix.lower().replace(' ', '_')}_score_hist.png"), dpi=150)
    plt.close()







    
        


# %%
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
"""
class SupConEncoder(nn.Module):
    def __init__(self, in_dim=5, hid=128, emb_dim=64, p_drop=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hid, hid),    nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.head = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, emb_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.head(h)
        z = F.normalize(z, dim=-1)  # L2-normalized embeddings
        return z

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z, y):
        # z: [B,D] (normalized), y: [B] int labels {0,1}
        B = z.size(0)
        sim = (z @ z.t()) / self.tau                      # [B,B]
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask.fill_diagonal_(False)                        # no self-contrast
        y = y.view(-1,1)
        pos_mask = (y == y.t()) & mask                    # positives by label
        # log-softmax over all non-self entries
        sim = sim.masked_fill(~mask, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # [B,B]
        # average over positives per anchor
        pos_count = pos_mask.sum(1).clamp_min(1)
        loss = -(log_prob.masked_select(pos_mask).sum() / pos_count.sum())
        return loss
"""

# %%
import torch, torch.nn as nn
import torch.nn.functional as F

class SupConEncoder(nn.Module):
    def __init__(self, in_dim, enc_dim=256, proj_dim=128):
        super().__init__()
        # backbone for tabular (use yours if you already have one)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, enc_dim), nn.ReLU(),
        )
        # projection head used ONLY for SupCon
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim, proj_dim),
        )

    def forward(self, x, *, return_proj=True):
        h = self.encoder(x)                 # [B, enc_dim]
        if not return_proj:
            return h
        z = self.proj(h)                    # [B, proj_dim]
        z = F.normalize(z, dim=-1)          # cosine space
        return z

# %%
import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, z, y):
        """
        z: [B, D] normalized embeddings
        y: [B] int labels (0/1)
        """
        z = F.normalize(z, dim=-1)                           # cosine
        logits = (z @ z.t()) / self.tau                      # [B,B]
        # mask out self-comparisons
        logits = logits - torch.eye(z.size(0), device=z.device) * 1e9

        # positive mask (same class & class==1 to avoid pulling negatives together)
        y = y.view(-1, 1)
        pos_mask = (y == y.t()) & (y == 1)

        # for each anchor, logsumexp over all others (denom)
        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)  # [B,1]

        # numerator: only positives
        # add -inf to non-positives so exp() zeroes them out
        num = torch.logsumexp(torch.where(pos_mask, logits, torch.full_like(logits, -1e9)), dim=1)

        # avoid anchors with no positives in batch
        valid = pos_mask.any(dim=1)
        loss = -(num[valid] - log_denom[valid].squeeze(1)).mean()
        return loss

# %% [markdown]
# 

# %%
def make_label_aware_batches(y_np, batch_size=2048, min_pos=128, seed=0):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y_np==1)[0]
    idx_neg = np.where(y_np==0)[0]
    while True:
        # sample positives with replacement if necessary
        p = rng.choice(idx_pos, size=min_pos, replace=(len(idx_pos) < min_pos))
        n = rng.choice(idx_neg, size=batch_size - min_pos, replace=False)
        batch_idx = np.concatenate([p, n])
        rng.shuffle(batch_idx)
        yield batch_idx

# %%
@torch.no_grad()
def pick_hard_negs_in_batch(z, y, max_negs_per_pos=4):
    # z: [B,D], y:[B], returns a boolean mask of which negatives to keep
    sim = z @ z.t()
    B = z.size(0)
    keep = torch.zeros(B, B, dtype=torch.bool, device=z.device)
    for i in range(B):
        pos = (y == y[i])
        neg = ~pos
        neg[i] = False
        # pick top hard negatives by similarity
        sim_i = sim[i].clone()
        sim_i[~neg] = -1e9
        idx = torch.topk(sim_i, k=min(max_negs_per_pos, neg.sum().item()))[1]
        keep[i, idx] = True
        # keep all positives for anchor i
        keep[i, pos & (torch.arange(B, device=z.device)!=i)] = True
    return keep  # use to mask logsumexp and pos sets if you want tighter mining

# %%
from sklearn.metrics import average_precision_score, roc_auc_score

def train_supcon(X_np, y_np, epochs=10, batch_size=2048, min_pos=128, lr=3e-4, wd=1e-4, tau=0.02, device="cuda"):
    model = SupConEncoder(in_dim=X_np.shape[1]).to(device)
    crit = SupConLoss(temperature=tau)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    sampler = make_label_aware_batches(y_np, batch_size=batch_size, min_pos=min_pos, seed=0)
    steps_per_epoch = max(1, (len(X_np)//batch_size))

    model.train()
    for ep in range(1, epochs+1):
        running = 0.0
        for _ in range(steps_per_epoch):
            bidx = next(sampler)
            xb = torch.from_numpy(X_np[bidx]).float().to(device)
            yb = torch.from_numpy(y_np[bidx]).long().to(device)
            z = model(xb)
            loss = crit(z, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        print(f"Epoch {ep:03d} | SupCon loss {running/steps_per_epoch:.4f}")
    model.eval()
    return model

# %%
from sklearn.neighbors import NearestNeighbors

@torch.no_grad()
def compute_embeddings(model, X_np, device="cuda", batch=65536):
    Z = []
    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(X_np[i:i+batch]).float().to(device)
        zb = model(xb).cpu().numpy()
        Z.append(zb)
    Z = np.vstack(Z)
    # already L2-normalized by model; normalize again for safety
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z

def knn_purity(Z, y_np, K=10, metric='cosine'):
    nn = NearestNeighbors(n_neighbors=K+1, metric=metric).fit(Z)
    d, idx = nn.kneighbors(Z, n_neighbors=K+1)   # includes self at [:,0]
    idx = idx[:,1:]                               # drop self
    nbr_labels = y_np[idx]                        # [N,K]
    # purity for positives only
    pos = (y_np==1)
    if pos.sum()==0: return 0.0, 0.0
    pos_purity = nbr_labels[pos].mean()           # fraction of positives among neighbors
    # probability of at least one positive neighbor
    p_at_least_one = 1.0 - (1.0 - nbr_labels[pos].mean(axis=1)).prod(axis=0)  # not exact; alternative below:
    # better: compute per-row: (nbr_labels[pos].sum(axis=1) > 0).mean()
    p_at_least_one = (nbr_labels[pos].sum(axis=1) > 0).mean()
    return float(pos_purity), float(p_at_least_one)

# %%
device = "cpu"  # or "cpu"
X_np, y_np = read_in_data()
model = train_supcon(
    X_np, y_np,
    epochs=10,
    batch_size=1000,   # lower on CPU (e.g., 512)
    min_pos=128,       # ensure your batch has enough positives
    lr=3e-4, wd=1e-4, tau=0.07,
    device=device
)

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

def knn_classifier_pr(Z, y_np, K=32, metric='cosine'):
    nn = NearestNeighbors(n_neighbors=K, metric=metric).fit(Z)
    distances, idx = nn.kneighbors(Z)
    preds = y_np[idx].mean(axis=1)  # proportion of positives among neighbors
    ap = average_precision_score(y_np, preds)
    roc = roc_auc_score(y_np, preds)
    return ap, roc

# %%
Z = compute_embeddings(model, X_np, device=device)  # (N, 64) L2-normalized

# Neighbor audit
pos_purity, p_at_least_one = knn_purity(Z, y_np, K=10, metric='cosine')
print(f"pos-purity@10={pos_purity:.3f} | P(≥1 pos in K=10)={p_at_least_one:.3f}")

# kNN classifier proxy
ap, roc = knn_classifier_pr(Z, y_np, K=32, metric='cosine')
print(f"kNN(PR-AUC)={ap:.4f} | kNN(ROC-AUC)={roc:.4f}")

# %%
sims = Z @ z_q
idx = sims.topk(k+1).indices
idx = idx[1:]  # remove self


