    def diffion(self, z_t, logit , lambda_diff=1e-3, k_diff=20, alpha_diff=1.0, t_diff=2):
        # ---- Diffusion consistency (per batch item) ----
        # L2-normalize latent so distances/margins are stable
        z_norm = F.normalize(z_t, p=2, dim=-1)

        L_diff = 0.0
        with torch.no_grad():
            # optional: detach P construction if you want only logit gradients;
            # remove detach if you want gradients through z as well.
            z_for_graph = z_norm.detach()

        for b in range(z_t.size(0)):  # loop over batch items
            z_b   = z_for_graph[b]             # (Nt, D)
            l_b   = logit[b]                   # (Nt,)
            P_b   = self.build_diffusion_P(z_b, k=k_diff, alpha=alpha_diff)   # (Nt, Nt)

            # t diffusion steps
            Ptb = P_b
            for _ in range(t_diff - 1):
                Ptb = Ptb @ P_b

            # diffuse logits (no grad through P to keep stable)
            l_smooth = (Ptb @ l_b)

            # consistency: encourage logits to be stable under local diffusion
            L_diff = L_diff + F.mse_loss(l_b, l_smooth)

            return lambda_diff * (L_diff / z_t.size(0))*1e20

    def _knn_mask_from_dists(self,D, k):
        # D: [N,N] squared distances; returns boolean mask keeping k neighbors (excl. self)
        N = D.size(0)
        D = D + torch.eye(N, device=D.device) * 1e6  # mask self
        idx = torch.topk(-D, k=k, dim=1).indices     # k nearest (largest negative distance)
        mask = torch.zeros_like(D, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    def build_diffusion_P(self,z, k=20, alpha=1.0, eps=None):
        """
        z: [N, D] L2-normalized latent (per sample or pooled query tokens)
        returns P: [N, N] row-stochastic diffusion operator with density correction.
        """
        # pairwise squared distances
        D2 = torch.cdist(z, z, p=2.0)**2  # [N,N]

        # bandwidth
        if eps is None:
            # distance to k-th neighbor per row, median over rows
            vals, _ = torch.topk(D2, k=k, dim=1, largest=False)
            eps = vals[:, -1].median().clamp_min(1e-6).detach()

        # raw kernel
        K = torch.exp(-D2 / eps)

        # sparsify with kNN (keeps graph local & cheap)
        knn_mask = self._knn_mask_from_dists(D2, k=k)
        K = K * knn_mask

        # density correction q_i
        q = K.sum(dim=1, keepdim=True).clamp_min(1e-12)          # [N,1]
        K = K / (q ** alpha)                                      # left
        K = K / (q.transpose(0,1) ** alpha)                       # right via outer; cheap approx

        # row normalize -> Markov
        P = K / K.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return P