import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
import torch.nn.functional as F

def supervised_contrastive_loss(z, y, temperature=0.1):
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

class SupervisedContrastive(nn.Module):
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
        supcon = supervised_contrastive_loss(z.squeeze(0), query_y.squeeze(0))*self.lambda_contrast
        

        return {"logits": [logits], "loss": supcon}

    def save(self,state, path):
        # drop all tree.leaf_cache.* entries from the state dict
        torch.save(state, path+'_model.pth')
    
    def load(self, path):
        state = torch.load(path+'_model.pth', map_location='cpu')
        self.load_state_dict(state['model_state'])

