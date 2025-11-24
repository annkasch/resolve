import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
import torch.nn.functional as F

def info_nce_loss(z: torch.Tensor,
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

class InfoNCE(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 d_model=64,
                 encoder_hidden=[128,128],
                 decoder_hidden=[128,128],
                 d_proj = 64,
                 lambda_contrast=0.01,
                 mode="concat",
                 theta_embed_dim=None):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        self.proj = MLP([d_model] + decoder_hidden + [d_proj])
        self.head = nn.Linear(d_proj, d_y)

    def forward(
        self,
        query_theta, query_phi, query_y,
        **_
    ):

        R_t = self.qry_enc(theta=query_theta, phi=query_phi)
        z = self.proj(R_t)
        logits = self.head(z)
        z = F.normalize(z, dim=-1)  
        supcon = info_nce_loss(z.squeeze(0), query_y.squeeze(0))*self.lambda_contrast
        

        return {"logits": [logits], "loss": supcon}


