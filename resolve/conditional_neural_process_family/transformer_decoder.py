import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.transformer_encoder import MLP, Attention, FeatureTokenizer

class CrossAttention(nn.Module):
    """Cross-attention: Q from target tokens, K/V from encoder representation."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate = torch.nn.Linear(dim // self.num_heads, 1)  # dim == D

    def forward(self, q_tokens, kv_tokens, key_mask, return_keys: bool=False, return_queries: bool=False):
        """
        q_tokens: (B, Nq, D)  - target tokens (queries)
        kv_tokens: (B, Nk, D) - encoder representation (keys/values)
        key_mask: (B, Nk) in {0,1}; 1 => POSITIVE key, 0 => NEGATIVE key
        """
        B, Nq, D = q_tokens.shape
        _, Nk, _ = kv_tokens.shape
        H = self.num_heads
        dh = D // H
        p_drop = self.attn_drop.p if self.training else 0.0

        # Project & split heads
        q = self.q(q_tokens).reshape(B, Nq, H, dh).permute(0, 2, 1, 3)  # (B,H,Nq,dh)
        k = self.k(kv_tokens).reshape(B, Nk, H, dh).permute(0, 2, 1, 3)  # (B,H,Nk,dh)
        v = self.v(kv_tokens).reshape(B, Nk, H, dh).permute(0, 2, 1, 3)  # (B,H,Nk,dh)

        # Build boolean keep masks per channel
        # keep_*: (B,1,1,Nk) -> broadcast to (B,H,Nq,Nk) for attn_mask
        keep_pos = (key_mask > 0)[:, None, None, :]         # True where POS
        keep_neg = ~keep_pos                                 # True where NEG

        # PyTorch SDPA: attn_mask=True means "mask out". So invert keep->mask.
        attn_mask_pos = ~keep_pos.expand(B, H, Nq, Nk)      # True = block
        attn_mask_neg = ~keep_neg.expand(B, H, Nq, Nk)

        # Handle edge cases: if a channel has no valid keys, skip SDPA to avoid NaNs
        has_pos = keep_pos.any(dim=-1).any(dim=-2)  # (B,1,1) -> per batch flag
        has_neg = keep_neg.any(dim=-1).any(dim=-2)

        # Pos channel
        if has_pos.any():
            x_pos = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask_pos, dropout_p=p_drop, is_causal=False
            )  # (B,H,Nq,dh)
        else:
            x_pos = torch.zeros((B, H, Nq, dh), device=q.device, dtype=q.dtype)

        # Neg channel
        if has_neg.any():
            x_neg = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask_neg, dropout_p=p_drop, is_causal=False
            )  # (B,H,Nq,dh)
        else:
            x_neg = torch.zeros((B, H, Nq, dh), device=q.device, dtype=q.dtype)

        # Learned gate lambda(q): sigmoid(linear per head/query over q)
        lam = torch.sigmoid(self.gate(q))  # (B,H,Nq,1)

        # Mix channels
        x_mix = lam * x_pos + (1.0 - lam) * x_neg  # (B,H,Nq,dh)

        # Merge heads and project out
        x = x_mix.transpose(1, 2).reshape(B, Nq, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_keys or return_queries:
            out = [x]
            if return_keys:
                # (B, Nk, D), labels per key from key_mask: 1=pos,0=neg
                out += [k.detach(), key_mask.detach().long()]
            if return_queries:
                # (B, Nq, D) — attach query labels if you have them
                out += [q.detach()]
            return tuple(out)

        return x

class TransformerDecoderBlock(nn.Module):
    """Self-attn on target tokens -> Cross-attn to encoder_repr -> MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm_ca_q = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x_tgt, encoder_tokens, wS):
        x_tgt = x_tgt + self.self_attn(self.norm_sa(x_tgt))
        x_tgt = x_tgt + self.cross_attn(self.norm_ca_q(x_tgt), encoder_tokens, wS)
        x_tgt = x_tgt + self.mlp(self.norm_mlp(x_tgt))
        return x_tgt

class TargetTransformerDecoder(nn.Module):
    """
    Transformer-like decoder for tabular targets with cross-attention to encoder outputs.
    Inputs:
      target_theta: (B, Nt, d_theta_t)
      target_phi:   (B, Nt, d_phi_t)
      encoder_repr: (B, Nc, D_enc)  # R_c from your ContextTransformerEncoder
    Output:
      y_hat:        (B, Nt, y_dim)
    """
    def __init__(
        self,
        target_theta_dim: int,
        target_phi_dim: int,
        y_dim: int,
        embed_dim: int = 128,
        enc_dim: int | None = None,     # D_enc; if None, assume equals embed_dim
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.F_total = target_theta_dim + target_phi_dim
        self.embed_dim = embed_dim

        self.tokenizer = FeatureTokenizer(self.F_total, embed_dim, bias=True, dropout=dropout)

        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.enc_dim = enc_dim if enc_dim is not None else embed_dim
        self.enc_proj = nn.Identity() if (self.enc_dim == embed_dim) else nn.Linear(self.enc_dim, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=dropout, attn_drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pred_head = nn.Sequential(nn.LayerNorm(embed_dim, eps=1e-6), nn.Linear(embed_dim, y_dim))

        self.apply(self._init_weights)

    def _aggregate(self, x):  # (B*, L, D)
        return x[:, 0, :] if self.use_cls else x.mean(dim=1)

    def forward(self, target_theta, target_phi, encoder_repr, context_y):
        """
        target_theta: (B, Nt, d_theta_t)
        target_phi:   (B, Nt, d_phi_t)
        encoder_repr: (B, Nc, D_enc)
        """
        B, Nt, _ = target_theta.shape
        Be, Nc, De = encoder_repr.shape
        assert B == Be, "Batch mismatch between targets and encoder_repr"

        feats = torch.cat([target_theta, target_phi], dim=-1).reshape(B * Nt, self.F_total)
        tgt_tokens = self.tokenizer(feats)  # (B*Nt, L_tgt, D)

        if self.use_cls:
            cls = self.cls_token.expand(tgt_tokens.shape[0], 1, self.embed_dim)
            tgt_tokens = torch.cat([cls, tgt_tokens], dim=1)

        wS = (context_y.squeeze(-1) > 0.5) 
        encoder = encoder_repr
        enc = self.enc_proj(encoder)     # (B, Nc, D)
        enc = enc.repeat_interleave(Nt, dim=0)  # (B*Nt, Nc, D)

        x = tgt_tokens
        for blk in self.blocks:
            x = blk(x, enc, wS)

        x = self.norm(x)
        h = self._aggregate(x)

        #y_hat = self.pred_head(h)
        return h.view(B, Nt, -1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)