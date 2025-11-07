import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.transformer_encoder import MLP, Attention, FeatureTokenizer


class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, model_dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q = nn.Linear(q_dim, model_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_dim, model_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_dim, model_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(model_dim, model_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, memory):
        B, Lq, _ = x.shape
        Bm, Lm, _ = memory.shape
        assert B == Bm, "Batch mismatch between x and memory"

        q = self.q(x).reshape(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(memory).reshape(B, Lm, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(memory).reshape(B, Lm, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, Lq, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DecoderBlock(nn.Module):
    """
    Self-attn -> (optional) cross-attn -> MLP.
    If memory is None, cross-attn is skipped (pure decoder without context).
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        mem_dim=None,
    ):
        super().__init__()
        mem_dim = mem_dim if mem_dim is not None else dim

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        # Uses your existing Attention class (self-attention without pos enc)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop)

        # Cross-attn pieces (used only if memory is provided at forward)
        self.norm2_x = nn.LayerNorm(dim, eps=1e-6)
        self.norm2_m = nn.LayerNorm(mem_dim, eps=1e-6)
        self.cross_attn = CrossAttention(q_dim=dim, kv_dim=mem_dim, model_dim=dim,
                                         num_heads=num_heads, qkv_bias=qkv_bias,
                                         attn_drop=attn_drop, proj_drop=drop)

        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x, memory=None):
        x = x + self.self_attn(self.norm1(x))

        if memory is not None:
            x_norm = self.norm2_x(x)
            m_norm = self.norm2_m(memory)
            x = x + self.cross_attn(x_norm, m_norm)

        x = x + self.mlp(self.norm3(x))
        return x


class ContextTransformerDecoder(nn.Module):
    """
    Target/tabular tokens -> stacked DecoderBlocks (+optional cross-attn to context) -> aggregated embedding.

    forward() supports three modes:
      1) Pure decoder (no context):            forward(target_feats)
      2) Cross-attn with precomputed memory:   forward(target_feats, memory=R_c)
      3) Cross-attn by calling encoder inline: forward(target_feats, encoder=enc,
                                                      context_theta=..., context_phi=..., context_y=...)
    """
    def __init__(
        self,
        target_feat_dim: int,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mem_dim: int | None = None,
        use_cls_token: bool = False,
        proj_out_dim: int | None = None
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.embed_dim = embed_dim
        self.F_total = target_feat_dim
        self.mem_dim = mem_dim if mem_dim is not None else embed_dim

        self.tokenizer = FeatureTokenizer(self.F_total, embed_dim, bias=True, dropout=dropout)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                mem_dim=self.mem_dim,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.Linear(embed_dim, proj_out_dim) if (proj_out_dim is not None) else nn.Identity()

        self.apply(self._init_weights)

    def _aggregate(self, x):  # (B, L, D)
        if self.use_cls:
            return x[:, 0, :]
        return x.mean(dim=1)

    @torch.no_grad()
    def _infer_mem_dim_from_encoder(self, encoder):
        # Best-effort: pull projected dim; if Identity, fall back to encoder.embed_dim
        if hasattr(encoder, "proj") and isinstance(encoder.proj, nn.Linear):
            return encoder.proj.out_features
        return getattr(encoder, "embed_dim", self.mem_dim)

    def forward(
        self,
        target_feats,                              # (B, Lt, Ft)
        memory: torch.Tensor | None = None,        # (B, Lm, Dm)
        *,
        encoder=None,                               # ContextTransformerEncoder (optional)
        context_theta=None, context_phi=None, context_y=None
    ):
        B, Lt, Ft = target_feats.shape
        assert Ft == self.F_total, f"target_feat_dim mismatch: expected {self.F_total}, got {Ft}"

        # If no memory is provided but an encoder + context are, compute memory on-the-fly.
        if memory is None and encoder is not None:
            # Optionally sync mem_dim to encoder output
            self.mem_dim = self._infer_mem_dim_from_encoder(encoder)
            memory = encoder(context_theta, context_phi, context_y)  # (B, Nc, Dm)

        # Tokenize target features per query
        feats = target_feats.reshape(B * Lt, Ft)
        tokens = self.tokenizer(feats)  # (B*Lt, Ft, D)

        if self.use_cls:
            cls = self.cls_token.expand(tokens.shape[0], 1, self.embed_dim)
            tokens = torch.cat([cls, tokens], dim=1)  # (B*Lt, 1+Ft, D)

        # Prepare memory for each query if present
        if memory is not None:
            mem_expanded = (
                memory.unsqueeze(1)
                .expand(B, Lt, memory.size(1), memory.size(2))
                .reshape(B * Lt, memory.size(1), memory.size(2))
            )
        else:
            mem_expanded = None

        for blk in self.blocks:
            tokens = blk(tokens, mem_expanded)

        tokens = self.norm(tokens)
        h = self._aggregate(tokens)          # (B*Lt, D)
        h = self.proj(h).view(B, Lt, -1)     # (B, Lt, D_out)
        return h

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)