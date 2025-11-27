import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Core transformer building blocks (single GPU, no positions)
# --------------------------

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    """Self-attention without positional encodings."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use PyTorch fused SDPA when available
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureTokenizer(nn.Module):
    """
    Per-scalar feature -> D-dim token, no positional encodings.
    """
    def __init__(self, num_features, embed_dim, bias=True, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, embed_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim)) if bias else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):     # x: (B, F)
        tok = x.unsqueeze(-1) * self.weight  # (B, F, D)
        if self.bias is not None:
            tok = tok + self.bias
        return self.drop(tok)


class TransformerEncoder(nn.Module):
    """
    Encodes (theta, phi, y) into per-context embeddings using a feature-token transformer.
    Shapes:
      theta: (B, Nc, d_theta)
      phi:   (B, Nc, d_phi)
      y:     (B, Nc, d_y)
    Returns:
      R_t: (B, Nc, D)
    """
    def __init__(
        self,
        theta_dim: int,
        phi_dim: int,
        y_dim: int | None = None,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        proj_out_dim: int | None = None,  # if set, final linear to this dim
        use_tokenizer: bool = False,
        use_cls_token: bool = False       # set False: mean over feature tokens
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.F_total = theta_dim + phi_dim 
        self.F_total += y_dim if (y_dim is not None) else 0
        self.embed_dim = embed_dim

        self.tokenizer = FeatureTokenizer(self.F_total, embed_dim, bias=True, dropout=dropout) if use_tokenizer else MLP(in_features=self.F_total,out_features=embed_dim)
        self.use_tokenizer = use_tokenizer
        
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=dropout, attn_drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.Linear(embed_dim, proj_out_dim) if (proj_out_dim is not None) else nn.Identity()
        
        self.apply(self._init_weights)

    def _aggregate(self, x):  # x: (B*, L, D)
        if self.use_cls:
            return x[:, 0, :]
        return x.mean(dim=1)

    def forward(self, theta, phi, y=None):
        B, Nc, _ = phi.shape
        # Concatenate features per context point: (B, Nc, F_total)
        feats = torch.cat([theta, phi, y], dim=-1) if y is not None else torch.cat([theta, phi], dim=-1)
        
        # Feature tokens (L = #features)
        if self.use_tokenizer: feats = feats.reshape(B * Nc, self.F_total)
        tokens = self.tokenizer(feats)

        if self.use_cls:
            cls = self.cls_token.expand(tokens.shape[0], 1, self.embed_dim)
            tokens = torch.cat([cls, tokens], dim=1)  # (B*Nc, 1+L, D)

        # Encoder
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        # Aggregate tokens -> per-context embedding
        h = tokens
        if self.use_tokenizer: h = self._aggregate(h)               # (B*Nc, D)
        h = self.proj(h).view(B, Nc, -1)          # (B, Nc, D_out)
        return h
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')