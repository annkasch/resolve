import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils ----------
class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, last_activation=False):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            is_last = (i == len(sizes) - 2)
            if (not is_last) or last_activation:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ThetaEncoder(nn.Module):
    """Encodes θ into an embedding."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, theta):
        leading = theta.shape[:-1]
        flat = theta.reshape(-1, theta.shape[-1])
        z = self.mlp(flat)
        return z.view(*leading, -1)

# context encoder: R^{(c,phi|θ)}
class ContextConditionalEncoder(nn.Module):
    """
    Encodes (context_theta, context_phi, context_y) -> R^(c)
    Shapes:
      context_theta: (B, Nc, d_theta)
      context_phi:   (B, Nc, d_phi)
      context_y:     (B, Nc, d_y)
      return:  (B, Nc, D)
    """
    def __init__(self,
                 phi_dim: int,
                 y_dim: int,
                 theta_in_dim: int,
                 theta_embed_dim: int,
                 hidden: list[int],
                 out_dim: int,
                 mode: str = "concat",
                 use_layernorm: bool = False):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in {"concat", "film"}, "mode must be 'concat' or 'film'"

        # θ encoder used only to condition the context features
        self.theta_enc = ThetaEncoder(theta_in_dim, [], theta_embed_dim)

        if self.mode == "concat":
            in_dim = phi_dim + y_dim + theta_embed_dim
            # allow empty hidden
            layers = [in_dim] + (hidden or []) + [out_dim]
            self.content = MLP(layers)
            self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

        else:  # FiLM
            # feature trunk on [phi,y]
            base_dims = [phi_dim + y_dim] + (hidden or [out_dim])
            self.feature_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            for i in range(len(base_dims) - 1):
                din, dout = base_dims[i], base_dims[i+1]
                self.feature_layers.append(nn.Linear(din, dout))
                self.film_layers.append(nn.Linear(theta_embed_dim, 2 * dout))
            # final projection if hidden was provided; if not, FiLM already outputs out_dim
            self.final = (
                nn.Linear(base_dims[-1], out_dim)
                if base_dims[-1] != out_dim
                else nn.Identity()
            )
            self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, context_phi, context_y, context_theta):
        B, Nc, _ = context_phi.shape
        theta_emb = self.theta_enc(context_theta)                       # (B,Nc,Eθ)

        if self.mode == "concat":
            x = torch.cat([context_phi, context_y, theta_emb], dim=-1)        # (B,Nc,·)
            out = self.content(x.view(B * Nc, -1)).view(B, Nc, -1)
            return self.norm(out)

        # FiLM path
        h = torch.cat([context_phi, context_y], dim=-1).view(B * Nc, -1)      # (B*Nc, d)
        cond = theta_emb.view(B * Nc, -1)                         # (B*Nc, Eθ)
        for lin, film in zip(self.feature_layers, self.film_layers):
            h = lin(h)
            gamma, beta = film(cond).chunk(2, dim=-1)            # (B*Nc, d)
            h = F.relu(gamma * h + beta)
        h = self.final(h)
        h = h.view(B, Nc, -1)
        return self.norm(h)
   
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

    def forward(self, Q_src, K_src, V_src=None, value_weights=None, mask=None):
        
        #Q_src: (B, Nt, D)  - from targets (R^{(t)})
        #K_src: (B, Nc, D)  - from context
        #V_src: (B, Nc, D)  - from context (defaults to K_src)
        #value_weights: (B, Nc) broadcast onto V

        if V_src is None: V_src = K_src

        Q = self._split(self.Wq(Q_src))          # (B,H,Nt,d_k)
        K = self._split(self.Wk(K_src))          # (B,H,Nc,d_k)
        V = self._split(self.Wv(V_src))          # (B,H,Nc,d_k)

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
        return self.out(context)                 # (B,Nt,D)

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
    
# ---------- target side ----------
class TargetQueryEncoder(nn.Module):
    """Builds R^{(t)} from (θ^t, φ^t)."""
    def __init__(self, theta_in_dim, phi_dim, hidden, out_dim, theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, query_theta, query_phi):
        theta_f = self.theta_enc(query_theta) if self.theta_enc else query_theta
        x = torch.cat([theta_f, query_phi], dim=-1)
        leading = x.shape[:-1]
        out = self.mlp(x.view(-1, x.shape[-1]))
        return out.view(*leading, -1)            # (B,Nt,D)

class TargetEncoder(nn.Module):
    """z_t = f([θ^t, φ^t, r^+_t, r^-_t])."""
    def __init__(self, theta_in_dim, phi_dim, r_dim, hidden, out_dim, theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim + 2*r_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, query_theta, query_phi, r_pos, r_neg):
        if r_pos.shape[1] == 1 and query_phi.shape[1] > 1:  # safety
            r_pos = r_pos.expand(-1, query_phi.shape[1], -1)
            r_neg = r_neg.expand_as(r_pos)
        theta_f = self.theta_enc(query_theta) if self.theta_enc else query_theta
        x = torch.cat([theta_f, query_phi, r_pos, r_neg], dim=-1)
        B, Nt, _ = x.shape
        z = self.mlp(x.view(B*Nt, -1))
        return z.view(B, Nt, -1)

class BernoulliHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [1])

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, 1)

# ---------- full model ----------
class HCTargetAttnNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=128):
        super().__init__()
        # Context side
        self.ctx_enc = ContextConditionalEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            theta_embed_dim=64, hidden=[128, 128], out_dim=d_model,
            mode='film', use_layernorm=True
        )

        self.theta_enc_t = ThetaEncoder(d_theta, [128, 128], d_model)
        
        # Target query R^{(t)}
        self.tquery = TargetQueryEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi,
            hidden=[128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Attention pooling (two passes with different value weights)
        #self.attn = GlobalContextAttention(d_model, n_heads=4, out_dim=d_model)
        self.attn = GlobalContextAttentionDual(d_model, n_heads=4, out_dim=d_model)
        
        # Target encoder -> z_t
        self.tgt_enc = TargetEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi, r_dim=d_model,
            hidden=[128, 128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Bernoulli decoder
        self.decoder = BernoulliHead(d_model)

    def forward(self, query_theta, query_phi, context_theta, context_phi, context_y, **kwargs):
        """
        Forward pass.

        Args:
            context_theta (Tensor): (B, Nc, d_theta)
            context_phi   (Tensor): (B, Nc, d_phi)
            context_y     (Tensor): (B, Nc, d_y)
            query_theta (Tensor): (B, Nt, d_theta)
            query_phi   (Tensor): (B, Nt, d_phi)

        Keyword Args:
            mask_c (Tensor, optional): (B, Nc) boolean mask for valid context rows.
        """
        mask_c = kwargs.get("mask_c", None)
        # in HCTargetAttnNP.forward(...)
        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=context_phi.device)

        # 1) Per-item context features
        R_ctx = self.ctx_enc(context_phi, context_y, context_theta)  # (B, Nc, D)

        # 2) Build per-target query R^{(t)}
        R_t = self.tquery(query_theta, query_phi)                 # (B,Nt,D)

        # 3) Class target attention pooling (two value-weighted passes)
        wS = context_y.squeeze(-1).clamp(0, 1)                  # (B,Nc)
        #wB = 1.0 - wS
        #r_pos = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wS, mask=mask_c)  # (B,Nt,D)
        #r_neg = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wB, mask=mask_c)  # (B,Nt,D)

        # single attention pass -> both r_pos and r_neg
        r_pos, r_neg = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, wS=wS, mask=mask_c)
        
        # 4) Target encoder -> z_t
        z_t = self.tgt_enc(query_theta, query_phi, r_pos, r_neg)  # (B,Nt,D)
        
        # 5) ComparatorHead: prototype interactions for sharper separation
        #feats = torch.cat([z_t, r_pos.expand_as(z_t), r_neg.expand_as(z_t), z_t*(r_pos-r_neg).expand_as(z_t)], dim=-1)
        # cosine similarity (B,Nt)
        #cos = torch.nn.functional.cosine_similarity(r_pos, r_neg, dim=-1)
        #print("cos mean:", cos.mean().item(), "cos p95:", cos.quantile(0.95).item())
        # relative L2 gap (B,Nt)
        #rel = (r_pos - r_neg).norm(dim=-1) / (0.5*(r_pos.norm(dim=-1)+r_neg.norm(dim=-1))+1e-8)
        #print("rel mean:", rel.mean().item(), "rel p95:", rel.quantile(0.95).item())
        
        # 6) Bernoulli decoder
        logit = self.decoder(z_t)                         # (B,Nt,1)

        output = {
            "logits": [logit]
        }
        
        return output