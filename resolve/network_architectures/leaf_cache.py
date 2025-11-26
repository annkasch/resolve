import torch
import torch.nn as nn

class LeafCache(nn.Module):
    """
    Generic memory bank for caching model outputs for global indices.

    - Stores scores and leaf embeddings for `num_samples` global items.
    - Uses a boolean mask to track which entries are filled.
    """

    def __init__(
        self,
        num_samples: int,
        out_dim: int,
        leaf_embed_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.num_samples = num_samples
        self.out_dim = out_dim
        self.leaf_embed_dim = leaf_embed_dim

        self.register_buffer(
            "score_bank",
            torch.zeros(num_samples, out_dim, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "leaf_bank",
            torch.zeros(num_samples, leaf_embed_dim, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "filled",
            torch.zeros(num_samples, dtype=torch.bool, device=device),
        )

    @torch.no_grad()
    def query(self, encoder, X: torch.Tensor, idx: torch.Tensor):
        """
        X:   (B,T,D) or (N,D)
        idx: same flattened length as X

        encoder: callable with signature encoder(X_sub) -> (scores_sub, leaf_sub)
        """
        if X.dim() == 3:
            B, T, D = X.shape
            X_flat = X.reshape(B * T, D)
        else:
            X_flat = X

        idx = idx.view(-1).to(self.filled.device)
        cached = self.filled[idx]
        need_compute = ~cached

        if need_compute.any():
            X_missing = X_flat[need_compute]
            scores_missing, leaf_missing = encoder(X_missing)

            if scores_missing.dim() == 1:
                scores_missing = scores_missing.unsqueeze(-1)

            assert scores_missing.shape[0] == leaf_missing.shape[0], \
                "Scores and leaf embeddings must have same batch size."

            idx_missing = idx[need_compute]
            
            self.score_bank[idx_missing] = scores_missing.to(self.score_bank.device)
            self.leaf_bank[idx_missing] = leaf_missing.to(self.leaf_bank.device)
            self.filled[idx_missing] = True

        scores = self.score_bank[idx]
        leaf = self.leaf_bank[idx]
        return scores, leaf
    
    def save_cache(self, path: str):
        payload = {
            "score_bank": self.score_bank.cpu(),
            "leaf_bank": self.leaf_bank.cpu(),
            "filled": self.filled.cpu(),
        }
        torch.save(payload, path)

    def load_cache(self, path: str, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        # basic sanity checks
        assert payload["score_bank"].shape == self.score_bank.shape
        assert payload["leaf_bank"].shape == self.leaf_bank.shape
        assert payload["filled"].shape == self.filled.shape

        self.score_bank.copy_(payload["score_bank"].to(self.score_bank.device))
        self.leaf_bank.copy_(payload["leaf_bank"].to(self.leaf_bank.device))
        self.filled.copy_(payload["filled"].to(self.filled.device))
    
