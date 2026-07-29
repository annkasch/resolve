from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.helpers import AsymmetricFocalWithFPPenalty, gaussian_nll, bce_with_logits
from sklearn.metrics import precision_recall_curve
import numpy as np
import os, faiss
from faiss.contrib import torch_utils
from typing import List, Optional, Sequence, Tuple, Dict, Union
from torch.utils.data import TensorDataset, DataLoader
from ..utilities import utilities as utils
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class MemoryBank:
    def __init__(self, d, encoder_online, encoder_momentum, tau=0.999, use_faiss=True,
                 use_gpu=True, ivf_nlist=None, fp16=True):
        self.N, self.d = 0, d
        self.online, self.momentum = encoder_online, encoder_momentum
        self.tau = tau
        self.E = None
        self.use_faiss = use_faiss
        self.use_gpu = use_gpu
        self.ivf_nlist = ivf_nlist
        self.fp16 = fp16
        self.index = None
        self.gpu_res = faiss.StandardGpuResources() if use_gpu else None
    
    def attach_dataset(self, dataset):
        self.dataset = InMemDataset(dataset)

    def fetch_by_ids(self, ids: torch.Tensor):
        # ids: 1D LongTensor (CPU)
        theta = self.dataset.theta[ids]
        phi = self.dataset.phi[ids]
        y = self.dataset.y[ids]
        # pull y only for context, not for queries
        return theta, phi, y
        #return theta.unsqueeze(0), phi.unsqueeze(0), self.dataset.data["train"]["y"][1][ids].unsqueeze(0) if hasattr(self.dataset.data["train"], "y") else None 

    def init_momentum_from_online(self):
        self.momentum.load_state_dict(self.online.state_dict())
        for p in self.momentum.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def ema_update(self):
        tau = self.tau
        for p_online, p_mom in zip(self.online.parameters(),
                                   self.momentum.parameters()):
            # p_mom = tau * p_mom + (1 - tau) * p_online
            p_mom.data.mul_(tau).add_(p_online.data, alpha=(1.0 - tau))

    @torch.no_grad()
    def build_from_loader(self, dataset, dtype=torch.float32):
        assert hasattr(self, "momentum")
        assert hasattr(self, "E")
        dataloader = dataset.set_loader(0, "train")
        self.N = len(dataloader.dataset)*dataloader.dataset.batch_size_tgt
        self.E = torch.empty(self.N, self.d, dtype=torch.float32)
        
        device = next(self.momentum.parameters()).device
        self.E = self.E.to(device)
        
        self.momentum.eval()
        torch.set_grad_enabled(False)

        with torch.amp.autocast(device.type, enabled=(device.type == "cuda")), torch.no_grad():
            for batch in dataloader:
                _, features,_ = batch
                theta = features.theta.to(device)
                phi = features.phi.to(device)

                # encode with momentum encoder
                idx = features.idx.to(device)
                hb = self.momentum(theta=theta, phi=phi)
                hb = torch.nn.functional.normalize(hb, dim=-1)
                hb = hb.to(dtype=dtype).contiguous()
                self.E[idx] = hb
    
    # ---------- Build bank from full tensors (GPU end-to-end) ----------
    @torch.no_grad()
    def build(self, batch_size: int = 16384):
        theta_all = self.dataset.theta
        phi_all = self.dataset.phi
        dev = next(self.momentum.parameters()).device
        self.momentum.eval()
        N = theta_all.size(0)
        self.N = N

        # Encode -> normalize on GPU; store bank on GPU for direct add()
        E = torch.empty((N, self.d), dtype=torch.float32, device=dev)
        use_amp = (dev.type == "cuda")
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            theta = theta_all[s:e].unsqueeze(0).to(dev, non_blocking=True)
            phi = phi_all[s:e].unsqueeze(0).to(dev, non_blocking=True)
            with torch.amp.autocast(dev.type, enabled=use_amp):
                h = self.momentum(theta=theta, phi=phi)     # (B, d)  NOTE: no unsqueeze!
            E[s:e] = F.normalize(h, dim=-1)

        self.E = E  # keep GPU tensor if you have VRAM; otherwise E.cpu()

        # ---- Build FAISS on GPU ----
        if not self.use_faiss:
            return self.E

        if self.use_gpu:
            if self.ivf_nlist is None:
                # Flat cosine on GPU (simple & fast)
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = bool(self.fp16)
                cfg.device = 0
                self.index = faiss.GpuIndexFlatIP(self.gpu_res, self.d, cfg)
                self.index.add(self.E)  # torch.cuda.FloatTensor accepted
            else:
                # IVF Flat on GPU: train on a subset
                X_cpu = self.E.float().cpu().numpy()  # train needs CPU index
                nlist = int(self.ivf_nlist)
                quant = faiss.IndexFlatIP(self.d)
                ivf = faiss.IndexIVFFlat(quant, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
                train_sz = min(200_000, X_cpu.shape[0])
                ivf.train(X_cpu[np.random.choice(X_cpu.shape[0], train_sz, replace=False)])
                # move trained IVF to GPU, add on GPU with torch tensor
                co = faiss.GpuClonerOptions()
                co.useFloat16 = bool(self.fp16)
                self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, ivf, co)
                self.index.nprobe = 16
                self.index.add(self.E)  # add torch CUDA tensor directly
        else:
            # CPU fallback (slower)
            X = self.E.float().cpu().numpy()
            faiss.normalize_L2(X)      # ensure unit length if using IP
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(X)


        return self.E
    
    @torch.no_grad()
    def _encode_targets_cuda(self, theta_t: torch.Tensor, phi_t: torch.Tensor) -> torch.Tensor:
        dev = next(self.momentum.parameters()).device
        with torch.amp.autocast(dev.type, enabled=(dev.type == "cuda")):
            z = self.momentum(theta=theta_t.to(dev, non_blocking=True),
                              phi=phi_t.to(dev,   non_blocking=True))  # (B,Nt,D)
        return F.normalize(z, dim=-1)  # keep on GPU

    @torch.no_grad()
    def topM_batch_with_llr(
        self,
        theta_t: torch.Tensor,
        phi_t: torch.Tensor,
        M: int = 32,
        self_ids: torch.Tensor | None = None,
        chunk: int = 8192,
        use_llr: bool = True,
        llr_weight: float = 1.0,
        expand_factor: int = 4,
    ):
        """
        FAISS retrieval with optional LLR-aware re-ranking.

        - First retrieves k = M * expand_factor (+1 if self_ids) candidates by similarity.
        - Then re-scores each candidate as: score = sim + llr_weight * llr[i]
        - Then keeps the top-M by this combined score (and optionally self-excludes).

        Assumes:
        self.index : FAISS index built on self.E (dimension d)
        self.llr   : (N,) tensor of LLR/logits aligned with memory indices.
        """
        assert self.index is not None, "FAISS index not built."
        z = self._encode_targets_cuda(theta_t, phi_t)   # (B,Nt,D), CUDA
        B, Nt, D = z.shape
        assert D == self.d
        Q = z.reshape(B * Nt, D).contiguous()           # (Q,D), CUDA

        # how many candidates per query we ask FAISS for
        k_base = M * (expand_factor if use_llr else 1)
        k = k_base + (1 if self_ids is not None else 0)

        # output containers (on GPU)
        all_I = torch.empty((B * Nt, M), dtype=torch.int64,   device=Q.device)
        all_S = torch.empty((B * Nt, M), dtype=torch.float32, device=Q.device)  # combined scores

        # bring LLR to CPU once for cheap indexing in Python loop
        llr_cpu = None
        if use_llr:
            llr_cpu = self.llr.detach().cpu().numpy()  # shape (N,)

        # self ids on CPU for easy comparison
        sid_cpu = None
        if self_ids is not None:
            sid_cpu = self_ids.reshape(-1).cpu().numpy()

        for s in range(0, Q.size(0), chunk):
            e = min(s + chunk, Q.size(0))

            # FAISS search -> similarity scores D_blk, indices I_blk (both on GPU)
            D_blk, I_blk = self.index.search(Q[s:e], k)   # (q_chunk, k)

            
            # move to CPU for flexible re-ranking logic
            I_cpu = I_blk.detach().cpu().numpy()
            D_cpu = D_blk.detach().cpu().numpy()

            # after D_blk, I_blk from FAISS
            labels_cand = self.dataset.y[I_cpu]              # (chunk, k)
            pos_frac_cand = (labels_cand > 0.5).float().mean().item()
            print("candidate pos fraction:", pos_frac_cand)

            print("sim range:", D_cpu.min().item(), D_cpu.max().item())
            print("llr range:", self.llr.min().item(), self.llr.max().item())


            out_i = []
            out_s = []

            
            for r in range(I_cpu.shape[0]):
                cand_idx = []
                cand_score = []

                # optional self-id for this query
                sid_r = sid_cpu[s + r] if sid_cpu is not None else None

                for idx_j, sim_j in zip(I_cpu[r], D_cpu[r]):
                    if idx_j < 0:
                        continue
                    # self-exclusion
                    if sid_r is not None and idx_j == int(sid_r):
                        continue

                    score_j = sim_j
                    if use_llr:
                        score_j += llr_weight * float(llr_cpu[idx_j])

                        

                    cand_idx.append(idx_j)
                    cand_score.append(score_j)

                # if nothing left, pad with -1
                if len(cand_idx) == 0:
                    cand_idx   = [-1] * M
                    cand_score = [-1.0] * M
                else:
                    # sort by combined score descending and keep top-M
                    order = np.argsort(cand_score)[::-1][:M]
                    cand_idx   = [cand_idx[j]   for j in order]
                    cand_score = [cand_score[j] for j in order]

                    # pad if fewer than M
                    if len(cand_idx) < M:
                        pad_n = M - len(cand_idx)
                        cand_idx   += [-1]   * pad_n
                        cand_score += [-1.0] * pad_n

                out_i.append(cand_idx)
                out_s.append(cand_score)

            # back to GPU
            all_I[s:e] = torch.tensor(out_i, device=Q.device, dtype=torch.int64)
            all_S[s:e] = torch.tensor(out_s, device=Q.device, dtype=torch.float32)

            # after re-ranking
            labels_top = self.dataset.y[all_I[s:e].detach().cpu()]
            pos_frac_top = (labels_top > 0.5).float().mean().item()
            print("top-M pos fraction:", pos_frac_top)

        ids  = all_I.view(B, Nt, M).cpu().numpy()
        sims = all_S.view(B, Nt, M).cpu().numpy()  # these are combined scores
        return ids, sims
    
    @torch.no_grad()
    def topM_batch(self, theta_t: torch.Tensor, phi_t: torch.Tensor,
                   M: int = 32, self_ids: torch.Tensor | None = None, chunk: int = 8192):
        assert self.index is not None, "FAISS index not built."
        z = self._encode_targets_cuda(theta_t, phi_t)         # (B,Nt,D), CUDA
        B, Nt, D = z.shape
        assert D == self.d
        Q = z.reshape(B * Nt, D).contiguous()                 # (Q,D), CUDA

        # Chunked GPU search; faiss accepts torch.cuda.FloatTensor via torch_utils
        all_I = torch.empty((B * Nt, M), dtype=torch.int64, device=Q.device)
        all_D = torch.empty((B * Nt, M), dtype=torch.float32, device=Q.device)
        k = M + (1 if self_ids is not None else 0)

        for s in range(0, Q.size(0), chunk):
            e = min(s + chunk, Q.size(0))
            D_blk, I_blk = self.index.search(Q[s:e], k)      # returns CUDA tensors
            if self_ids is None:
                all_I[s:e] = I_blk[:, :M]
                all_D[s:e] = D_blk[:, :M]
            else:
                sid = self_ids.reshape(-1)[s:e].to(Q.device)
                # self-exclude on GPU
                mask = (I_blk != sid.unsqueeze(1))
                # take first M true per row
                # simple fallback to CPU for selection if you prefer:
                I_cpu = I_blk.detach().cpu().numpy()
                D_cpu = D_blk.detach().cpu().numpy()
                out_i = []
                out_d = []
                for r in range(I_cpu.shape[0]):
                    keep = [(i, d) for i, d in zip(I_cpu[r], D_cpu[r]) if i != int(sid[r - 0].item())]
                    keep = keep[:M]
                    if len(keep) < M:
                        keep += [(-1, -1.0)] * (M - len(keep))
                    ii, dd = zip(*keep)
                    out_i.append(ii); out_d.append(dd)
                all_I[s:e] = torch.tensor(out_i, device=Q.device, dtype=torch.int64)
                all_D[s:e] = torch.tensor(out_d, device=Q.device, dtype=torch.float32)

        ids  = all_I.view(B, Nt, M).cpu().numpy()
        sims = all_D.view(B, Nt, M).cpu().numpy()
        return ids, sims
    
    def supervised_contrastive_loss(self, z, y, temperature=0.1):
        """
        z: (B, d) normalized
        y: (B,) int {0,1}
        """
        B = z.size(0)
        sim = z @ z.t() / temperature          # (B, B)
        # mask out self
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, -1e9)

        # positive mask: same label, excluding self
        y = y.view(-1, 1)
        pos_mask = (y == y.t()) & (~self_mask)   # (B, B)

        # for each anchor i, positives P(i) = j where pos_mask[i,j] = True
        # numerator: sum_j∈P(i) exp(sim_ij)
        exp_sim = torch.exp(sim)                 # (B,B)

        pos_exp = exp_sim * pos_mask            # zero out non-positives
        pos_sum = pos_exp.sum(dim=1)            # (B,)

        # denominator: sum over all j != i
        all_sum = exp_sim.sum(dim=1)            # (B,)

        # only consider anchors with at least one positive
        valid = pos_sum > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # SupCon per anchor: -log( sum_pos / sum_all )
        loss_i = -torch.log(pos_sum[valid] / all_sum[valid])
        return loss_i.mean()
    
    def pairwise_ranking_loss(self,logits, y, margin=1.0):
        """
        logits: (B,)
        y:      (B,) in {0,1}
        """
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return logits.new_tensor(0.0)

        pos_logits = logits[pos_mask]   # (P,)
        neg_logits = logits[neg_mask]   # (N,)

        # all pairwise differences: f(p) - f(n)
        diff = pos_logits[:, None] - neg_logits[None, :]   # (P, N)

        # hinge: max(0, margin - (f(p) - f(n)))
        loss_mat = F.relu(margin - diff)
        return loss_mat.mean()


    def llr_contrastive_loss(self, logits: torch.Tensor,
                            y: torch.Tensor,
                            margin: float = 0.0,
                            max_pairs: int | None = 10_000) -> torch.Tensor:
        """
        LLR-aware contrastive loss (pairwise ranking on logits).

        logits: (B,) raw scores from your density head (before sigmoid)
        y:      (B,) labels in {0,1}
        margin: optional margin m; we enforce l_pos - l_neg >= m
        max_pairs: cap the number of pairs for speed (None = all pairs)
        """
        device = logits.device
        y = y.view(-1).bool()
        pos_logits = logits[y]          # (P,)
        neg_logits = logits[~y]         # (N,)

        P = pos_logits.size(0)
        N = neg_logits.size(0)

        if P == 0 or N == 0:
            return torch.tensor(0.0, device=device)

        # all pairwise differences: (P, N)
        diff = pos_logits.view(P, 1) - neg_logits.view(1, N)  # l_pos - l_neg

        # flatten to (P*N,)
        diff = diff.view(-1)

        # subsample pairs if very large (for speed)
        if max_pairs is not None and diff.numel() > max_pairs:
            idx = torch.randint(0, diff.numel(), (max_pairs,), device=device)
            diff = diff[idx]

        # apply margin: want diff >= margin
        # loss = softplus(margin - diff) = log(1 + exp(margin - (l_pos - l_neg)))
        loss = F.softplus(margin - diff).mean()
        return loss

    def info_nce_loss(self,z: torch.Tensor,
                  y: torch.Tensor,
                  tau: float = 0.1) -> torch.Tensor:
        """
        Supervised InfoNCE / NT-Xent-style loss (vectorized).

        z:  (B, D) embeddings
        y:  (B,) integer labels (e.g. 0/1)
        tau: temperature
        """
        device = z.device
        B = z.size(0)

        # normalize for cosine similarity
        z = F.normalize(z, dim=-1)

        # similarity matrix: (B, B)
        sim = torch.matmul(z, z.T) / tau

        # mask out self-similarity
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e9)

        # positive mask: same label, excluding self
        y = y.view(-1, 1)                          # (B,1)
        pos_mask = (y == y.T) & (~self_mask)       # (B,B) bool

        # for numerical stability: log-softmax over all others (as denominator)
        # log_prob_ij = log exp(sim_ij) - log sum_k exp(sim_ik)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (B,B)

        # we want for each anchor i: average over its positives:
        # L_i = - (1/|P(i)|) sum_{j in P(i)} log_prob_ij
        # where P(i) are indices j with same label as i (and j != i)

        # sum log_prob over positives for each anchor
        pos_log_prob = (log_prob * pos_mask).sum(dim=1)        # (B,)

        # how many positives per anchor?
        pos_counts = pos_mask.sum(dim=1)                       # (B,)

        # avoid division by zero: only anchors with at least one positive contribute
        valid = pos_counts > 0
        if valid.sum() == 0:
            # no usable anchors in this batch
            return torch.tensor(0.0, device=device)

        # average log prob over positives
        mean_pos_log_prob = pos_log_prob[valid] / pos_counts[valid]

        # InfoNCE loss: negative of that
        loss = -mean_pos_log_prob.mean()
        return loss

    def train_density_ratio_head(self, epochs: int = 5, batch_size: int = 4096,
                                 lr: float = 1e-3, prior: float | None = None, writer = None,
                                 device: torch.device | None = None):
        """
        Train a small head f(E[i]) -> logit P(y=1|E[i]) on the fixed embeddings self.E.

        Must be called AFTER self.build() and AFTER self.attach_dataset().
        """
        assert self.E is not None, "Call build() before training density ratio head."
        assert hasattr(self, "dataset"), "Attach dataset to MemoryBank first."

        dev = device or self.E.device
        E = self.E.detach().to(dev)        # (N, d), fixed features

        # labels: assume dataset.data["train"]["y"][1] is (N,)
        y_all = self.dataset.y
        y_all = torch.as_tensor(y_all, dtype=torch.float32, device=dev).view(-1)
        assert E.size(0) == y_all.size(0), "Embedding/label length mismatch."

        N, D = E.shape

        # tiny MLP head: logits f(h)
        class DensityRatioModule(nn.Module):
            def __init__(self, d_in, proj_dim=32):
                super().__init__()
                # shared trunk
                self.proj = nn.Sequential(
                    nn.Linear(d_in, 128),
                    nn.ReLU(),
                    nn.Linear(128, proj_dim)
                )
                # density-ratio head on top of proj space
                self.head = nn.Linear(proj_dim, 1)  # logits

            def forward(self, h):
                # h: (B, d_in)
                z = self.proj(h)                    # (B, proj_dim)
                z = F.normalize(z, dim=-1)          # important for contrastive
                logits = self.head(z).squeeze(-1)   # (B,)
                return logits, z

        self.dr_head = DensityRatioModule(D).to(dev)

        # imbalance handling: BCE with pos_weight
        pi_emp = float(y_all.mean().item()) if y_all.sum() > 0 else 1e-6

        pos_weight = (1.0 - pi_emp) / max(pi_emp, 1e-6)

        #criterion = nn.BCEWithLogitsLoss(
        #    pos_weight=torch.tensor(pos_weight, device=dev)
        #)
        criterion = AsymmetricFocalWithFPPenalty(
                            alpha_pos=0.9315,
                            alpha_neg=None,
                            gamma_pos=2.,
                            gamma_neg=2.,
                            lambda_fp=0.,
                            tau_fp=0.5,
                            lambda_tp= 0.,
                            tau_tp=0.5,
                            reduction="mean",
                            base_loss_fn=bce_with_logits,
                    )
        optimizer = torch.optim.Adam(self.dr_head.parameters(), lr=lr)

        # DataLoader over fixed embeddings
        ds = TensorDataset(E, y_all)
        #loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        weights = torch.where(y_all == 1,
                      torch.tensor(10.0),
                      torch.tensor(1.0))
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(y_all),
            replacement=True,
        )
        loader = DataLoader(ds, batch_size=256, sampler=sampler)
        y_true_batch = []
        y_pred_batch = []
        self.dr_head.train()
        for ep in range(epochs):
            pbar = tqdm(loader, total=len(loader), desc="llr head", leave=True)
            for h_batch, y_batch in pbar:
                logits,z = self.dr_head(h_batch)              # (B,)
                loss = criterion(logits, y_batch)
                lambda_contrast = 0.01  # tune
                #supcon = self.supervised_contrastive_loss(z, y_batch)
                #info = self.info_nce_loss(z, y_batch) 
                #print(loss.item(), supcon)
                #loss += lambda_contrast * supcon
                llr_rank = self.llr_contrastive_loss(logits, y_batch, margin=0.1)
                #loss_rank  = self.pairwise_ranking_loss(logits, y_batch, margin=1.0)
                #print(loss.item(), lambda_contrast*llr_rank.item())
                loss += lambda_contrast * llr_rank

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                y_true_batch.append(y_batch)
                y_pred_batch.append(torch.sigmoid(logits))
                #pbar.set_postfix(loss=f"{running_loss/len(y_true_all):.4f}")
            
            y_true = torch.cat(y_true_batch).float().detach().cpu().numpy()
            y_pred = torch.cat(y_pred_batch).float().detach().cpu().numpy()

            fig = utils.plot(y_pred.reshape(-1, 1), y_true.reshape(-1, 1), it=ep+1)
            writer.add_figure(f'plot/score_llr_head', fig, global_step=ep+1)
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            fig = plt.figure()
            plt.plot(precision,recall)
            plt.xlabel("Signal Efficiency (Recall)")
            plt.ylabel("Precision")
            writer.add_figure(f'plot/prec_recall_llr', fig, global_step=ep+1)
            # optional: print / log
            # print(f"[DR] epoch {ep}: loss={loss.item():.4f}")

            # ---- Precompute LLR for all N samples ----
            self.dr_head.eval()
            all_logits = []
            with torch.no_grad():
                for h_batch, _ in DataLoader(ds, batch_size=batch_size, shuffle=False):
                    logits,z = self.dr_head(h_batch.to(dev))
                    all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=0)  # (N,)
            assert all_logits.size(0) == N

            # posterior log-odds (logit) minus prior odds -> LLR
            if prior is None:
                pi = pi_emp  # use empirical prior by default
            else:
                pi = float(prior)
            log_prior_odds = math.log(pi) - math.log(1.0 - pi)

            self.llr = all_logits - log_prior_odds  # store on CPU
            print("mean llr pos:", self.llr[y_all==1].mean())
            print("mean llr neg:", self.llr[y_all==0].mean())

class InMemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dict):
        self.theta = dataset["theta"]
        self.phi   = dataset["phi"]
        self.y     = dataset["y"]

    def __len__(self):
        return self.theta.size(0)

    def __getitem__(self, idx):
        yield self.theta[idx], self.phi[idx], self.y[idx]
