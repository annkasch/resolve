from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os, faiss
os.environ["OMP_NUM_THREADS"] = "4"
faiss.omp_set_num_threads(4)

class MemoryBank:
    def __init__(self, d, encoder_online, encoder_momentum, tau=0.999, use_faiss=True):
        self.N, self.d = 0, d
        self.online, self.momentum = encoder_online, encoder_momentum
        self.tau = tau
        self.E = None
        self.use_faiss = use_faiss
        if use_faiss:
            self.index = faiss.IndexFlatIP(d)

    @torch.no_grad()
    def build_from_loader(self, dataset, dtype=torch.float32):
        assert hasattr(self, "momentum")
        assert hasattr(self, "E")
        dataloader = dataset.set_loader("train")
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
    
    @torch.no_grad()
    def build(self, theta_all: torch.Tensor, phi_all: torch.Tensor,
                batch_size: int = 8192):
        """
        Build memory bank from full tensors in RAM.
        Assumes: bank row i == dataset id i (recommended).
        Stores bank on CPU float32; builds FAISS IndexFlatIP with cosine.
        """
        device = next(self.momentum.parameters()).device
        self.momentum.eval()
        N = theta_all.size(0)

        self.N = N
        self.E = torch.empty((N, self.d), dtype=torch.float32, device="cpu")

        use_amp = (device.type == "cuda")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            theta = theta_all[start:end].to(device, non_blocking=True)
            phi = phi_all[start:end].to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp), torch.no_grad():
                h = self.momentum(theta=theta.unsqueeze(0), phi=phi.unsqueeze(0))              # (B, d)
            h = torch.nn.functional.normalize(h, dim=-1).float().cpu()

            self.E[start:end] = h

        # Build FAISS (cosine via inner product on L2-normalized vectors)
        if self.use_faiss:
            X = self.E.numpy().astype("float32", copy=False)
            if X.size == 0:
                raise ValueError("Empty embedding bank (N=0).")
            X = np.ascontiguousarray(X)                # ensure C-order
            if not np.isfinite(X).all():
                raise ValueError("Embeddings contain NaN/Inf.")

            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(norms, 1e-12, None)        # avoid divide-by-zero
            self.index = faiss.IndexFlatIP(self.d)
            self.index.add(X)

        return self.E
    
    @torch.no_grad()
    def _encode_targets(self, theta_t: torch.Tensor, phi_t: torch.Tensor) -> torch.Tensor:
        """
        Encode targets with the momentum encoder (inputs-only) and L2-normalize.
        Input:  theta_t, phi_t with shape (B, Nt, …)
        Output: z with shape (B, Nt, D), float32 (on CPU)
        """
        device = next(self.momentum.parameters()).device
        use_amp = (device.type == "cuda")

        with torch.amp.autocast(device.type, enabled=use_amp):
            z = self.momentum(theta=theta_t.to(device, non_blocking=True),
                              phi=phi_t.to(device,   non_blocking=True))  # (B,Nt,D)
        z = F.normalize(z, dim=-1).float().cpu()  # unit vectors for cosine/IP
        return z  # (B,Nt,D) CPU tensor

    @torch.no_grad()
    def topM_batch(self,
                   theta_t: torch.Tensor,
                   phi_t: torch.Tensor,
                   M: int = 128,
                   self_ids: torch.Tensor | None = None):
        """
        Batched retrieval with optional self-exclusion.

        Args:
            theta_t, phi_t: (B, Nt, …) target inputs
            M:              top-M neighbors per target
            self_ids:       optional (B, Nt) LongTensor of dataset ids to exclude

        Returns:
            ids:  (B, Nt, M) np.int64  — neighbor indices (bank rows; = dataset ids if you built 1:1)
            sims: (B, Nt, M) np.float32 — similarity scores (cosine if IndexFlatIP + unit-norm)
        """
        assert self.index is not None, "FAISS index not built."
        z = self._encode_targets(theta_t, phi_t)  # (B,Nt,D) CPU float32
        B, Nt, D = z.shape
        assert D == self.d, f"Dim mismatch: query D={D}, index d={self.d}"

        # Flatten to (B*Nt, D) for FAISS
        q = z.reshape(B * Nt, D).contiguous().numpy().astype("float32", copy=False)
        
        ids_flat, sims_flat = self.faiss_search_chunked(self.index, q, M, self_ids=None, chunk=100)
        ids  = ids_flat.reshape(B, Nt, M)
        sims = sims_flat.reshape(B, Nt, M)
        return ids, sims

    def faiss_search_chunked(self, index, q, M, self_ids=None, chunk=100):

        Bn, d = q.shape
        K = int(M + (1 if self_ids is not None else 0))
        K = min(K, getattr(index, "ntotal", K))      # cap to ntotal

        all_I = np.empty((Bn, M), dtype=np.int64)
        all_D = np.empty((Bn, M), dtype=np.float32)

        for s in range(0, Bn, chunk):
            print("1",s, chunk)
            e = min(s + chunk, Bn)
            D_blk, I_blk = index.search(q[s:e], K)   # (blk, K)
            print("2",s, chunk)

            if self_ids is None:
                all_I[s:e] = I_blk[:, :M]
                all_D[s:e] = D_blk[:, :M].astype(np.float32, copy=False)
            else:
                sid = np.asarray(self_ids[s:e], dtype=np.int64)
                out_i = np.empty((e - s, M), dtype=I_blk.dtype)
                out_d = np.empty((e - s, M), dtype=D_blk.dtype)
                for r in range(e - s):
                    keep_i, keep_d = [], []
                    for i, d in zip(I_blk[r], D_blk[r]):
                        if i == sid[r]:        # self-exclude
                            continue
                        keep_i.append(i); keep_d.append(d)
                        if len(keep_i) == M: break
                    # pad if fewer than M (rare)
                    if len(keep_i) < M:
                        pad = M - len(keep_i)
                        keep_i += [-1]*pad
                        keep_d += [np.float32(-1)]*pad
                    out_i[r], out_d[r] = keep_i, np.asarray(keep_d, dtype=np.float32)
                all_I[s:e], all_D[s:e] = out_i, out_d
        return all_I, all_D