import torch
import torch.nn as nn
import torch.nn.functional as F

# You already have these in your codebase; importing to stay consistent.
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP

# -----------------------
# helpers
# -----------------------
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1):
    """mask: bool True=keep, False=mask-out."""
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    return torch.softmax(logits, dim=dim)

def build_knn_mask(emb: torch.Tensor, k: int, mask_valid: torch.Tensor):
    """
    emb: (B, Nc, D)    context embeddings
    mask_valid: (B, Nc)  True=valid context row
    returns: knn_mask (B, Nc, Nc) boolean with True for (i,j) if j in top-k of i (self included)
    """
    B, Nc, D = emb.shape
    # cosine similarity for stability
    e = F.normalize(emb, dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", e, e)  # (B,Nc,Nc)

    # forbid invalid nodes as neighbors
    valid = mask_valid.unsqueeze(1).expand(B, Nc, Nc) & mask_valid.unsqueeze(2).expand(B, Nc, Nc)
    sim = torch.where(valid, sim, torch.full_like(sim, -1e9))

    # include self
    eye = torch.eye(Nc, device=emb.device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)

    # top-k per row (boost self slightly to guarantee inclusion)
    sim_plus = sim + eye.float() * 1e-3
    topk_idx = sim_plus.topk(k=min(k, Nc), dim=-1).indices  # (B,Nc,k)

    knn_mask = torch.zeros(B, Nc, Nc, device=emb.device, dtype=torch.bool)
    knn_mask.scatter_(dim=2, index=topk_idx, value=True)
    knn_mask = knn_mask | eye
    knn_mask = knn_mask & valid
    return knn_mask  # (B,Nc,Nc)


# -----------------------
# Local Graph Attention (single layer, kNN-masked)
# -----------------------
class LocalGraphAttn(nn.Module):
    """
    One-hop graph attention over the retrieved context set.
    Attention is computed with learned projections and masked to kNN neighbors.
    """
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.h = heads
        self.dh = d_model // heads
        self.scale = self.dh ** -0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, Rc: torch.Tensor, mask_knn: torch.Tensor):
        """
        Rc: (B, Nc, D)   context embeddings
        mask_knn: (B, Nc, Nc) bool, True if j is neighbor of i
        returns: Rc_refined (B, Nc, D)
        """
        B, Nc, D = Rc.shape
        q = self.Wq(Rc).view(B, Nc, self.h, self.dh)   # (B,Nc,H,Dh)
        k = self.Wk(Rc).view(B, Nc, self.h, self.dh)
        v = self.Wv(Rc).view(B, Nc, self.h, self.dh)

        # logits over neighbors
        logits = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale   # (B,Nc,Nc,H)

        # apply kNN mask (shared across heads)
        mask = mask_knn.unsqueeze(-1).expand_as(logits)               # (B,Nc,Nc,H)
        att = masked_softmax(logits, mask, dim=2)                     # softmax over j

        # aggregate
        msg = torch.einsum("bijh,bjhd->bihd", att, v)                 # (B,Nc,H,Dh)
        msg = msg.reshape(B, Nc, D)
        msg = self.out(msg)
        return self.ln(Rc + self.dropout(msg))                        # residual + norm


# -----------------------
# Target→Context Cross-Attention (aggregator: r_t = Σ α h_i)
# -----------------------
class TargetCrossAttention(nn.Module):
    """
    Scaled dot-product cross-attention: query at targets, keys/values at refined context.
    """
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dh = d_model // heads
        self.scale = self.dh ** -0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_c = nn.LayerNorm(d_model)

    def forward(self, Rt: torch.Tensor, Rc_refined: torch.Tensor, mask_c: torch.Tensor):
        """
        Rt: (B, Nt, D) targets (query side)
        Rc_refined: (B, Nc, D) context (key/value side)
        mask_c: (B, Nc) bool, True=keep context row
        returns: r_t (B, Nt, D), attn_weights (B, Nt, Nc)
        """
        B, Nt, D = Rt.shape
        Nc = Rc_refined.size(1)
        q = self.Wq(self.ln_q(Rt)).view(B, Nt, self.h, self.dh)      # (B,Nt,H,Dh)
        k = self.Wk(self.ln_c(Rc_refined)).view(B, Nc, self.h, self.dh)
        v = self.Wv(self.ln_c(Rc_refined)).view(B, Nc, self.h, self.dh)

        # logits_{t,i,h} = <q_t^h, k_i^h> / sqrt(Dh)
        logits = torch.einsum("bthd,bihd->btih", q, k) * self.scale  # (B,Nt,Nc,H)

        # mask invalid context rows
        mask = mask_c.unsqueeze(1).unsqueeze(-1).expand_as(logits)   # (B,Nt,Nc,H)
        att = masked_softmax(logits, mask, dim=2)                    # softmax over Nc

        # attended value per target/head
        z = torch.einsum("btih,bihd->bthd", att, v)                  # (B,Nt,H,Dh)
        z = z.reshape(B, Nt, D)
        z = self.out(z)
        r_t = z                                                     # aggregated vector per target
        return r_t, att.mean(dim=-1)  # (B,Nt,D), (B,Nt,Nc)


# -----------------------
# Prediction head
# -----------------------
class Head(nn.Module):
    def __init__(self, d_in: int, hidden=[256, 256], out_dim: int = 1):
        super().__init__()
        self.mlp = MLP([d_in] + hidden + [out_dim])

    def forward(self, x):
        return self.mlp(x)


# -----------------------
# TCL-CNP (Local GAT + Cross-Attn Aggregator)
# -----------------------
class TCL_CNP_LocalGAT(nn.Module):
    """
    Encoder → (retrieved context) → Local kNN GAT (context-only)
           → Target→Context Cross-Attention (aggregator r_t)
           → Head → logits

    forward(...) returns {"logits": (B,Nt,1), "aux": {...}}
    """
    def __init__(self,
                 d_theta, d_phi, d_y,
                 d_model=128,
                 encoder_hidden=[128, 128],
                 mode="concat",
                 theta_embed_dim=None,
                 k_local=20,
                 heads_local=4,
                 heads_cross=4,
                 dropout=0.0,
                 head_hidden=[256, 256],
                 concat_rt=True  # feed [R_t || r_t] to head (often helps)
                 ):
        super().__init__()
        self.d_model = d_model
        self.k_local = k_local
        self.concat_rt = concat_rt

        # Context encoder: uses (theta_c, phi_c, y_c)
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        # Target query encoder: uses (theta_t, phi_t) – no y_t at inference
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        # Local graph attention over retrieved context
        self.local_gat = LocalGraphAttn(d_model=d_model, heads=heads_local, dropout=dropout)

        # Target→Context cross-attention (aggregator)
        self.cross = TargetCrossAttention(d_model=d_model, heads=heads_cross, dropout=dropout)

        # Prediction head
        in_dim = (2 * d_model) if concat_rt else d_model
        self.head = Head(d_in=in_dim, hidden=head_hidden, out_dim=1)

        # Norms
        self.norm_ctx_out = nn.LayerNorm(d_model)
        self.norm_rt = nn.LayerNorm(d_model)

    def forward(
        self,
        query_theta, query_phi, query_y,      # query_y kept for API; not used
        context_theta, context_phi, context_y,
        *,
        mask_c=None,                          # (B,Nc) bool; True=valid context row
        train=True,
        **kwargs
    ):
        """
        IMPORTANT: context_* must already be the retrieved top-M set for each target batch.
        Shapes:
            context_theta: (B, Nc, d_theta)
            context_phi  : (B, Nc, d_phi)
            context_y    : (B, Nc, d_y)
            query_theta  : (B, Nt, d_theta)
            query_phi    : (B, Nt, d_phi)
        """
        device = context_phi.device
        B, Nc, _ = context_phi.shape
        Nt = query_phi.size(1)
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=device)

        # 1) Encode
        Rc0 = self.ctx_enc(theta=context_theta, phi=context_phi, y=context_y)  # (B,Nc,D)
        Rt  = self.qry_enc(theta=query_theta,  phi=query_phi)                  # (B,Nt,D)

        # 2) Local Graph Attention over context (kNN inside retrieved set)
        knn_mask = build_knn_mask(Rc0, k=self.k_local, mask_valid=mask_c)      # (B,Nc,Nc)
        Rc = self.local_gat(Rc0, knn_mask)                                     # (B,Nc,D)
        Rc = self.norm_ctx_out(Rc)

        # 3) Target→Context Cross-Attention (aggregator r_t = Σ α h_i)
        r_t, attn_tc = self.cross(Rt, Rc, mask_c=mask_c)                       # (B,Nt,D), (B,Nt,Nc)
        r_t = self.norm_rt(r_t)

        # 4) Prediction Head
        dec_in = torch.cat([Rt, r_t], dim=-1) if self.concat_rt else r_t       # (B,Nt, *)
        logits = self.head(dec_in)                                             # (B,Nt,1)

        aux = {
            "attn_tc": attn_tc,     # (B,Nt,Nc)
            "Rc0": Rc0,             # (B,Nc,D) pre-GAT
            "Rc": Rc,               # (B,Nc,D) post-GAT
            "Rt": Rt,               # (B,Nt,D)
            "knn_mask": knn_mask    # (B,Nc,Nc)
        }
        return {"logits": logits, "aux": aux}