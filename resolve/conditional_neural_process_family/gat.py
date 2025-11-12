import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Basic MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, dims, act=nn.GELU):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2:
                layers += [act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Graph Attention Layer (GAT)
# -------------------------------
class GATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads)**-0.5

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, knn_idx):
        """
        x:     (B, N, D)
        knn_idx: (B, N, K) indices of neighbors
        """
        B, N, D = x.shape
        K = knn_idx.size(-1)
        H = self.heads
        Dh = D // H

        q = self.W_q(x).view(B, N, H, Dh)
        k = self.W_k(x).view(B, N, H, Dh)
        v = self.W_v(x).view(B, N, H, Dh)

        # gather neighbors
        knn_k = torch.gather(k, 1, knn_idx.unsqueeze(-1).expand(B, N, K, H, Dh))
        knn_v = torch.gather(v, 1, knn_idx.unsqueeze(-1).expand(B, N, K, H, Dh))

        # compute attention scores
        q_i = q.unsqueeze(2)                     # (B,N,1,H,Dh)
        att = (q_i * knn_k).sum(-1) * self.scale # (B,N,K,H)
        att = F.softmax(att, dim=2)

        # weighted sum of v_j
        out = (att.unsqueeze(-1) * knn_v).sum(2) # (B,N,H,Dh)
        out = out.reshape(B, N, D)

        return self.out(out)


# -------------------------------
# Graph-CNP (no pooling)
# -------------------------------
class GraphCNP(nn.Module):
    def __init__(self,
                 d_theta,
                 d_phi,
                 d_y,
                 d_model=64,
                 encoder_hidden=[128,128],
                 k=30,
                 gat_layers=4,
                 heads=4):
        super().__init__()

        self.k = k

        # Encode each item independently
        in_dim = d_theta + d_phi + d_y
        self.enc = MLP([in_dim] + encoder_hidden + [d_model])

        # GAT stack
        self.gnn = nn.ModuleList([
            GATLayer(d_model, heads=heads) for _ in range(gat_layers)
        ])

        # Final per-item decoder (logits)
        self.dec = MLP([d_model] + [d_model, 1])

    # --------- kNN graph construction ----------
    def compute_knn(self, x, k):
        """
        x: (B,N,D)
        return knn_idx: (B,N,k)
        """
        B, N, D = x.shape
        # pairwise distances
        dist = torch.cdist(x, x)               # (B,N,N)
        knn = dist.topk(k=k, largest=False).indices
        return knn

    # -------------------------------------------
    def forward(self, query_theta, query_phi, query_y,
                      context_theta, context_phi, context_y,
                      train=True, **kwargs):

        # Combine context + target into one big set
        # (because graph is built over all)
        x_theta = torch.cat([context_theta, query_theta], dim=1)
        x_phi   = torch.cat([context_phi,   query_phi],   dim=1)
        x_y     = torch.cat([context_y,     query_y],     dim=1)

        B, N, _ = x_theta.shape

        # per item encoding
        feats = torch.cat([x_theta, x_phi, x_y], dim=-1)
        h = self.enc(feats)                    # (B,N,D)

        # kNN graph
        knn_idx = self.compute_knn(h, self.k)

        # GAT propagation
        for gat in self.gnn:
            h = h + gat(h, knn_idx)           # residual GAT

        # decode per-item logit
        logits_all = self.dec(h)              # (B,N,1)

        # split out target logits only
        Nc = context_theta.size(1)
        logits_t = logits_all[:, Nc:, :]

        return {"logits": logits_t}