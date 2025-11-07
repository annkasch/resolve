import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- attention pooling (per target) ----------
class GlobalContextAttention(nn.Module):
    
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

    def attention(self, Q_src, K_src, V_src=None, value_weights=None, mask=None):
        
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
        return self.out(context), Wk, Wv                 # (B,Nt,D)
    
    def forward(self, Q_src, K_src_pos, K_src_neg, V_src_pos=None, value_weights_pos=None, V_src_neg=None, value_weights_neg=None ):
        r_pos, Wk_pos, Wv_pos = self.attention(Q_src, K_src_pos, V_src=None, value_weights=None)
        r_neg, Wk_neg, Wv_neg = self.attention(Q_src, K_src_neg, V_src=None, value_weights=None)

        loss = self.l_proj_param(Wk_pos[0], Wk_neg[0], Wv_pos[0], Wv_neg[0], normalize=True, eps=1e-12)
        return r_pos, r_neg, loss




    def l_proj_param(self, Wk_pos, Wk_neg, Wv_pos, Wv_neg, normalize=True, eps=1e-12):
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

class GlobalContextAttentionDual(nn.Module):
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

    def forward(self, Q_src, K_src, V_src, wS, mask=None):
        """
        Q_src: (B, Nt, D)
        K_src: (B, Nc, D)
        V_src: (B, Nc, D)
        wS:    (B, Nc)    -- positive-class per-key weights in [0,1]
        mask:  (B, Nc) bool, True for VALID keys
        Returns: (r_pos, r_neg) each (B, Nt, D_out)
        """
        B, Nt, _ = Q_src.shape
        _, Nc, _ = K_src.shape

        Q = self._split(self.Wq(Q_src))   # (B,H,Nt,d_k)
        K = self._split(self.Wk(K_src))   # (B,H,Nc,d_k)
        V = self._split(self.Wv(V_src))   # (B,H,Nc,d_k)

        # scores and softmax (one time)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,H,Nt,Nc)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)  # (B,H,Nt,Nc)

        # two value matmuls: all and weighted
        V_all = V
        V_pos = V * wS[:, None, :, None]      # broadcast (B,1,Nc,1)

        context_all = torch.matmul(attn, V_all)     # (B,H,Nt,d_k)
        context_pos = torch.matmul(attn, V_pos)     # (B,H,Nt,d_k)
        context_neg = context_all - context_pos                 # uses wB = 1 - wS

        r_pos = self.out(self._merge(context_pos))  # (B,Nt,D)
        r_neg = self.out(self._merge(context_neg))  # (B,Nt,D)
        return r_pos, r_neg