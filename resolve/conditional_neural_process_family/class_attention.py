import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def l_proj_param(Wk_pos, Wk_neg, Wv_pos, Wv_neg, normalize=True, eps=1e-12):
    # W*: [d_k, d_in] as in nn.Linear(out=d_k, in=d_in).weight
    def cross_gram(Wa, Wb):
        return Wa @ Wb.t()  # [d_k, d_k]

    Gk = cross_gram(Wk_pos, Wk_neg)
    Gv = cross_gram(Wv_pos, Wv_neg)

    if normalize:
        nk = (Wk_pos.norm(p='fro') * Wk_neg.norm(p='fro')).clamp_min(eps)
        nv = (Wv_pos.norm(p='fro') * Wv_neg.norm(p='fro')).clamp_min(eps)
        return (Gk.pow(2).sum() / nk) + (Gv.pow(2).sum() / nv)
    else:
        return Gk.pow(2).sum() + Gv.pow(2).sum()

def l_proj_param_batched(Wk_pos, Wk_neg, Wv_pos, Wv_neg, normalize=True, eps=1e-12):
    """
    All inputs: (B*H, d_k, Nc). Returns scalar loss averaged over B*H.
    """
    # Cross-grams: (B*H, d_k, d_k)
    Gk = torch.bmm(Wk_pos, Wk_neg.transpose(1, 2))
    Gv = torch.bmm(Wv_pos, Wv_neg.transpose(1, 2))

    if normalize:
        nk = (Wk_pos.norm(dim=(1, 2)) * Wk_neg.norm(dim=(1, 2))).clamp_min(eps)  # (B*H,)
        nv = (Wv_pos.norm(dim=(1, 2)) * Wv_neg.norm(dim=(1, 2))).clamp_min(eps)
        loss_k = (Gk.pow(2).sum(dim=(1, 2)) / nk)  # (B*H,)
        loss_v = (Gv.pow(2).sum(dim=(1, 2)) / nv)
        return (loss_k + loss_v).mean()
    else:
        return (Gk.pow(2).sum(dim=(1, 2)) + Gv.pow(2).sum(dim=(1, 2))).mean()



# ---------- attention pooling (per target) ----------
class CrossAttention(nn.Module):
    
    def __init__(self, d_model, n_heads=4, out_dim=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, out_dim or d_model)

    def _split(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,N,d_k)

    def forward(self, Q_src, K_src, V_src=None, value_weights=None, mask=None):
        
        #Q_src: (B, Nt, D)  - from targets (R^{(t)})
        #K_src: (B, Nc, D)  - from context
        #V_src: (B, Nc, D)  - from context (defaults to K_src)
        #value_weights: (B, Nc) broadcast onto V

        if V_src is None: V_src = K_src
        Wk = self.Wk(K_src)
        Wv = self.Wv(V_src)

        Q = self._split(self.Wq(Q_src))          # (B,H,Nt,d_k)
        K = self._split(Wk)          # (B,H,Nc,d_k)
        V = self._split(Wv)          # (B,H,Nc,d_k)

        # scores and softmax (one time)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,H,Nt,Nc)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        if value_weights is not None:
            w = value_weights[:, None, :, None]  # (B,1,1,Nc,1)
            V = V * w
        context = torch.matmul(attn, V)          # (B,H,Nt,d_k)

        B, H, Nt, d_k = context.shape
        context = context.transpose(1, 2).contiguous().view(B, Nt, H*d_k)
        return self.out(context)

class CrossAttentionWithMoE(nn.Module):
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

class CrossAttentionDual(nn.Module):
    """
    Multi-head attention that returns (context_pos, context_neg) given a single
    per-key weight vector wS in [0,1]. Computes scores/softmax once, then
    uses linearity: attn@V and attn@(V*wS); neg = all - pos.
    """
    def __init__(self, d_model, n_heads=4, out_dim=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, out_dim or d_model)

    def _split(self, x):
        # (B,N,D) -> (B,H,N,d_k)
        B, N, D = x.shape
        return x.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        # (B,H,N,d_k) -> (B,N,D)
        B, H, N, d_k = x.shape
        return x.transpose(1, 2).reshape(B, N, H * d_k)

    def forward(self, Q_src, K_src, wS_ctx, V_src=None, mask=None):
        """
        Q_src: (B, Nt, D)
        K_src: (B, Nc, D)
        V_src: (B, Nc, D)
        wS:    (B, Nc)    -- positive-class per-key weights in [0,1]
        mask:  (B, Nc) bool, True for VALID keys
        Returns: (r_pos, r_neg) each (B, Nt, D_out)
        """
            
        B, Nc, Dk = K_src.shape

        # Multi-head split
        def _split(x, H):
            B_, N_, D_ = x.shape
            dk = D_ // H
            return x.view(B_, N_, H, dk).permute(0, 2, 1, 3)           # (B,H,N,dk)

        if V_src is None: V_src = K_src
        Q = _split(self.Wq(Q_src), self.n_heads)                               # (B,H,Nt,dk)
        K = _split(self.Wk(K_src), self.n_heads)                             # (B,H,Nc,dk)
        V = _split(self.Wv(V_src), self.n_heads)                             # (B,H,Nc,dk)

        # Shared scores once
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Dk ** 0.5)    # (B,H,Nt,Nc)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn_all = torch.softmax(scores, dim=-1)                       # shared attention (B,H,Nt,Nc)

        # Build class-renormalized attentions (no leakage)
        # wS for forward path: hard/soft labels but NO epsilon smoothing
        wS = wS_ctx.float().clamp(0., 1.)                              # (B,Nc)
        wS_attn = wS[:, None, None, :]                                 # (B,1,1,Nc)

        attn_pos = attn_all * wS_attn
        Zp = attn_pos.sum(-1, keepdim=True).clamp_min(1e-12)
        attn_pos = attn_pos / Zp

        attn_neg = attn_all * (1.0 - wS_attn)
        Zn = attn_neg.sum(-1, keepdim=True).clamp_min(1e-12)
        attn_neg = attn_neg / Zn

        # Contexts
        C_all  = torch.matmul(attn_all, V)                             # (B,H,Nt,dk)
        C_pos  = torch.matmul(attn_pos, V)                             # (B,H,Nt,dk)
        C_neg  = torch.matmul(attn_neg, V)                             # (B,H,Nt,dk)

        # Signed comparison channel (no renorm; captures margin in same neighborhood)
        s_signed = (2.0 * wS - 1.0)[:, None, None, :]                  # (B,1,1,Nc)
        C_diff   = torch.matmul(attn_all * s_signed, V)                # (B,H,Nt,dk)

        # Merge heads + output projection
        def _merge(x):
            B_, H_, Nt_, dk_ = x.shape
            return x.permute(0, 2, 1, 3).contiguous().view(B_, Nt_, H_ * dk_)

        r_all  = self.out(_merge(C_all))                               # (B,Nt,D)
        r_pos  = self.out(_merge(C_pos))                               # (B,Nt,D)
        r_neg  = self.out(_merge(C_neg))                               # (B,Nt,D)
        r_diff = self.out(_merge(C_diff))                              # (B,Nt,D)


        return r_all, r_pos, r_neg, r_diff



class SimplePoolAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                Q_src: torch.Tensor,      # (B, Nt, D)
                K_src: torch.Tensor,      # (B, Nc, D)
                V_src: torch.Tensor | None = None,  # (B, Nc, Dv) or None -> use K
                mask: torch.Tensor | None = None,   # (B, Nc) bool
                logit_bias_ctx: torch.Tensor | None = None,  # (B, Nc)
                beta: float = 1.0):
        """
        Q_src: queries (R_t)
        K_src: keys   (R_ctx)
        V_src: values (usually same as K_src or another projection)
        mask:  True for valid context positions
        logit_bias_ctx: per-context log-bias (e.g. LLR)
        """
        B, Nt, D = Q_src.shape
        _, Nc, Dk = K_src.shape
        assert D == Dk, "Q and K must have same dim"

        if V_src is None:
            V_src = K_src

        # 1) similarity scores
        # scores: (B, Nt, Nc)
        scores = torch.einsum("bqd,bkd->bqk", Q_src, K_src) / math.sqrt(D)

        # 2) add LLR / density-ratio bias per context, if provided
        if logit_bias_ctx is not None:
            # logit_bias_ctx: (B, Nc)
            scores = scores + beta * logit_bias_ctx.unsqueeze(1)  # -> (B, Nt, Nc)

        # 3) mask invalid contexts, if any
        if mask is not None:
            # mask: (B, Nc) -> (B, 1, Nc)
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        # 4) softmax over contexts
        attn = F.softmax(scores, dim=-1)  # (B, Nt, Nc)

        # 5) weighted sum of values
        # out: (B, Nt, Dv)
        out = torch.einsum("bqk,bkd->bqd", attn, V_src)

        return out, attn