import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
in_slurm = "SLURM_JOB_ID" in os.environ
from resolve.network_architectures.normalizing_flow import NVPFlow

class TensorDataset1D(Dataset):
    def __init__(self, theta, phi):
        """
        x: torch.Tensor of shape (N, D)
        """
        self.theta = theta
        self.phi = phi

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):
        return self.theta[idx], self.phi[idx]

class NormalizingFlowClassifier(nn.Module):
    def __init__(self, dim, n_flow_layers=6, hidden_dims=[128, 128]):
        super().__init__()
        self.dim = dim
        self.pos_flow = NVPFlow(dim, n_flow_layers, hidden_dims)
        self.neg_flow = NVPFlow(dim, n_flow_layers, hidden_dims)
        self.pi_pos = 0.5
        self.pi_neg = 1. - self.pi_pos

    def forward(self, query_theta, query_phi, **kwargs):
        with torch.no_grad():
            logp_pos = self.pos_flow(query_theta, query_phi)["log_prob"]
            logp_neg = self.neg_flow(query_theta, query_phi)["log_prob"]
        
        log_likelihood_ratio = logp_pos - logp_neg
        logit = log_likelihood_ratio + math.log(self.pi_pos) - math.log(self.pi_neg)
        logit = torch.sigmoid(logit)

        loss = -log_likelihood_ratio.mean()

        out = {"logits": [logit], "loss": loss}
        return out
    
    def fit(
        self,
        trainer,
        **kwargs,
    ):

        loader = trainer.dataset.set_loader(0, "train")
        batch_size = loader.dataset.batch_size
        theta_neg, phi_neg, _ = loader.dataset.get_negatives("train")
        n_neg = phi_neg.shape[0]
        ds_train_neg = TensorDataset1D(theta_neg, phi_neg)
        train_loader = DataLoader(ds_train_neg, batch_size=batch_size, shuffle=True)
        self.train_realnvp_flow(self.neg_flow, train_loader, trainer.device)

        theta_pos, phi_pos, _ = loader.dataset.get_positives("train")
        n_pos = phi_pos.shape[0]
        ds_train_pos = TensorDataset1D(theta_pos, phi_pos)
        train_loader = DataLoader(ds_train_pos, batch_size=batch_size, shuffle=True)
        self.train_realnvp_flow(self.pos_flow, train_loader, trainer.device)
        
        self.pi_pos = n_pos / (n_pos + n_neg)
        self.pi_neg = 1. - self.pi_pos

        trainer.num_eposchs = 1

    def train_realnvp_flow(self, flow, train_loader, device):
        optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

        # ---- train ----
        self.neg_flow.train()
        train_nll_sum = 0.0
        train_n = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"train flow", leave=True, disable=False)

        for i, batch in enumerate(pbar):
            theta, phi = batch
            optimizer.zero_grad()

            nll = flow(query_theta=theta.to(device), query_phi=phi.to(device))["loss"]  # (B,

            nll.backward()
            optimizer.step()

            train_nll_sum += nll.item() * phi.shape[-2]
            train_n += phi.shape[-2]
            pbar.set_postfix(loss=f"{train_nll_sum/train_n:.4f}")

        avg_train_nll = train_nll_sum / train_n

    def save(self,state, path):
        # drop all tree.leaf_cache.* entries from the state dict
        torch.save(state, path+'_model.pth')
    
    def load(self, path):
        state = torch.load(path+'_model.pth', map_location='cpu')
        self.load_state_dict(state['model_state'])