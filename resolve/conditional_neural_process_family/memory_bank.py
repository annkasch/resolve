import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    θ-aware EMA prototype memory with separate positive/negative slots.
    - keys are L2-normalized embeddings in R^D (same D as your ctx/tgt features)
    - per θ-cell we keep up to J_pos positive and J_neg negative prototypes
    """
    def __init__(self, num_theta_cells: int, dim: int, J_pos: int = 5, J_neg: int = 10,
                 alpha: float = 0.99, tau_assign: float = 0.5, device="cpu"):
        super().__init__()
        self.C = num_theta_cells
        self.D = dim
        self.J_pos = J_pos
        self.J_neg = J_neg
        self.alpha = alpha
        self.tau_assign = tau_assign
        #self.device = device

        # Prototypes (C, J, D); valid counts per cell
        self.register_buffer("pos", torch.zeros(self.C, self.J_pos, self.D, device=device))
        self.register_buffer("neg", torch.zeros(self.C, self.J_neg, self.D, device=device))
        self.register_buffer("pos_mask", torch.zeros(self.C, self.J_pos, dtype=torch.bool, device=device))
        self.register_buffer("neg_mask", torch.zeros(self.C, self.J_neg, dtype=torch.bool, device=device))

    @torch.no_grad()
    def _assign_and_update(self, table, mask, k, cell, is_pos: bool):
        """
        table: (C,J,D), mask: (C,J)
        k: (D,), cell: int
        """

        entries = table[cell]                                  # (J,D)
        valid = mask[cell]                                     # (J,)

        if valid.any():
            sims = torch.mv(entries[valid], k)                 # (J_valid,)
            j_local = torch.argmax(sims).item()
            best_sim = sims[j_local].item()
            j = torch.arange(valid.numel(), device=valid.device)[valid][j_local].item()
        else:
            best_sim, j = -1.0, None

        # create new if no valid or too dissimilar and capacity available
        if (not valid.any() or best_sim < self.tau_assign) and valid.sum().item() < entries.size(0):
            j_new = torch.argmax(~valid).item()               # first free slot
            table[cell, j_new] = F.normalize(k, dim=0)
            #mask[cell, j_new] = True
            return
        # otherwise EMA-update nearest
        if j is not None:
            proto = table[cell, j]
            proto.copy_(F.normalize(self.alpha * proto + (1 - self.alpha) * k, dim=0))

    @torch.no_grad()
    def write(self, k: torch.Tensor, theta_cell: torch.Tensor, y: torch.Tensor, is_hard_neg: torch.Tensor=None):
        """
        k: (B,N,D) normalized keys to write (teacher/stable encoder)
        theta_cell: (B,N) long ids in [0,C)
        y: (B,N,1) labels in {0,1} or probabilities
        is_hard_neg: (B,N,1) bool (optional) – marks negatives to store
        """
        B, N, D = k.shape
        assert D == self.D
        kf = k.reshape(B*N, D)
        cells = theta_cell.reshape(B*N).long()

        ys = y.reshape(B*N, -1)

        if is_hard_neg is None:
            is_hard_neg = torch.zeros_like(ys, dtype=torch.bool)
        else:
            is_hard_neg = is_hard_neg.reshape(B*N, -1).bool()
        
        for i in range(B*N):
            cell = int(cells[i].item())

            if cell < 0 or cell >= self.C:  # skip invalid cell ids
                continue
            key = kf[i]

            yi = float(ys[i].item())
            if yi >= 0.5:
                self._assign_and_update(self.pos, self.pos_mask, key, cell, True)
            #elif is_hard_neg[i].item():
            #    self._assign_and_update(self.neg, self.neg_mask, key, cell, False)


    def _gather_cell_protos(self, table, mask, cells):
        # cells: (B,Nt) long -> list of (B,Nt,J,D) with invalid entries masked later
        B, Nt = cells.shape
        J = table.size(1)
        out = table[cells]             # (B,Nt,J,D)
        msk = mask[cells]              # (B,Nt,J)
        return out, msk

    @torch.no_grad()
    def read(self, q: torch.Tensor, qry_theta_cell: torch.Tensor,
            K_pos: int = 4, K_neg: int = 0, tau: float = 0.1, lambda_neg: float = 0.5):
        """
        Memory-efficient read:
        - loops over unique θ cells present in the batch,
        - no (B,Nt,J,D) gathers,
        - only small (M,J,D) matrices live briefly.

        q: (B,Nt,D) normalized
        qry_theta_cell: (B,Nt) long
        returns: r_mem, r_pos, r_neg each (B,Nt,D)
        """
        dev = q.device
        B, Nt, D = q.shape
        r_pos = torch.zeros(B, Nt, D, device=dev)
        r_neg = torch.zeros(B, Nt, D, device=dev)

        cells_flat = qry_theta_cell.view(-1).long()
        uniq_cells = torch.unique(cells_flat)

        q_flat = q.view(B * Nt, D)
        rpos_flat = r_pos.view(B * Nt, D)
        rneg_flat = r_neg.view(B * Nt, D)

        
        for c in uniq_cells.tolist():
            idx = (cells_flat == c).nonzero(as_tuple=False).squeeze(1)  # (M,)
            if idx.numel() == 0:
                continue

            # ---- positives ----
            mpos = self.pos_mask[c]  # (Jp,)
            if mpos.any():
                P = self.pos[c, mpos]                    # (Jp,D)
                sims = (q_flat[idx] @ P.t()) / tau       # (M,Jp)
                k = min(K_pos, sims.size(1))
                vals, ids = torch.topk(sims, k=k, dim=1)
                w = torch.softmax(vals, dim=1)           # (M,k)
                P_sel = P[ids]                           # (M,k,D)
                rpos_flat[idx] = (w.unsqueeze(-1) * P_sel).sum(1)

            
            # ---- negatives (optional) ----
            if K_neg > 0:
                mneg = self.neg_mask[c]
                if mneg.any():
                    N = self.neg[c, mneg]                # (Jn,D)
                    sims_n = (q_flat[idx] @ N.t()) / tau
                    k = min(K_neg, sims_n.size(1))
                    v, idn = torch.topk(sims_n, k=k, dim=1)
                    w = torch.softmax(v, dim=1)
                    N_sel = N[idn]
                    rneg_flat[idx] = (w.unsqueeze(-1) * N_sel).sum(1)

        r_pos = rpos_flat.view(B, Nt, D)
        r_neg = rneg_flat.view(B, Nt, D)
        r_mem = r_pos - lambda_neg * r_neg
        return r_mem, r_pos, r_neg