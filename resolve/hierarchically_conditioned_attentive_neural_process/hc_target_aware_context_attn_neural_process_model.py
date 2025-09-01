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

# ---------- context encoder: R^{(c,phi|θ)} ----------
class ContextConditionalEncoder(nn.Module):
    """
    Encodes (theta_c, phi_c, y_c) -> R^(c)
    Shapes:
      theta_c: (B, Nc, d_theta)
      phi_c:   (B, Nc, d_phi)
      y_c:     (B, Nc, d_y)
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

    def forward(self, phi_c, y_c, theta_c):
        B, Nc, _ = phi_c.shape
        theta_emb = self.theta_enc(theta_c)                       # (B,Nc,Eθ)

        if self.mode == "concat":
            x = torch.cat([phi_c, y_c, theta_emb], dim=-1)        # (B,Nc,·)
            out = self.content(x.view(B * Nc, -1)).view(B, Nc, -1)
            return self.norm(out)

        # FiLM path
        h = torch.cat([phi_c, y_c], dim=-1).view(B * Nc, -1)      # (B*Nc, d)
        cond = theta_emb.view(B * Nc, -1)                         # (B*Nc, Eθ)
        for lin, film in zip(self.feature_layers, self.film_layers):
            h = lin(h)
            gamma, beta = film(cond).chunk(2, dim=-1)            # (B*Nc, d)
            h = F.relu(gamma * h + beta)
        h = self.final(h)
        h = h.view(B, Nc, -1)
        return self.norm(h)

class MAB(nn.Module):
    """
    Multihead Attention Block (Set Transformer).
    Pre-norm + residual: x -> LN -> MHA -> + -> LN -> FF -> +
    Supports key_padding_mask (B, N) where True = pad/invalid.
    """
    def __init__(self, d_model, n_heads=4, ff_hidden=256, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, key_padding_mask=None, need_weights=False):
        # Pre-norm
        q = self.ln1(Q)
        k = self.ln1(K)
        v = self.ln1(V)
        # nn.MultiheadAttention expects True for pads in key_padding_mask
        # key_padding_mask: (B, N_k) with True for "ignore"
        attn_out, attn_weights = self.attn(
            q, k, v, key_padding_mask=key_padding_mask, need_weights=need_weights
        )
        x = Q + self.dropout(attn_out)
        y = x + self.dropout(self.ff(self.ln2(x)))
        return y, (attn_weights if need_weights else None)

class SAB(nn.Module):
    """
    Stack of L self-attention (MAB) layers over a set.
    """
    def __init__(self, d_model, n_heads=4, ff_hidden=256, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            MAB(d_model, n_heads=n_heads, ff_hidden=ff_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, key_padding_mask=None, return_attn=False):
        attns = []
        for layer in self.layers:
            X, attn_w = layer(X, X, X, key_padding_mask=key_padding_mask, need_weights=return_attn)
            if return_attn:
                attns.append(attn_w)
        return (X, attns) if return_attn else X
    
# ---------- attention pooling (per target) ----------
class GlobalContextAttention(nn.Module):
    """Multi-head attention with optional value weights."""
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
        """
        Q_src: (B, Nt, D)  - from targets (R^{(t)})
        K_src: (B, Nc, D)  - from context
        V_src: (B, Nc, D)  - from context (defaults to K_src)
        value_weights: (B, Nc) broadcast onto V
        """
        if V_src is None: V_src = K_src

        Q = self._split(self.Wq(Q_src))          # (B,H,Nt,d_k)
        K = self._split(self.Wk(K_src))          # (B,H,Nc,d_k)
        V = self._split(self.Wv(V_src))          # (B,H,Nc,d_k)

        if value_weights is not None:
            w = value_weights[:, None, :, None]  # (B,1,1,Nc,1)
            V = V * w

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,H,Nt,Nc)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)          # (B,H,Nt,d_k)

        B, H, Nt, d_k = context.shape
        context = context.transpose(1, 2).contiguous().view(B, Nt, H*d_k)
        return self.out(context)                 # (B,Nt,D)

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

    def forward(self, theta_t, phi_t):
        theta_f = self.theta_enc(theta_t) if self.theta_enc else theta_t
        x = torch.cat([theta_f, phi_t], dim=-1)
        leading = x.shape[:-1]
        out = self.mlp(x.view(-1, x.shape[-1]))
        return out.view(*leading, -1)            # (B,Nt,D)

class FactorizedPairwiseKV(nn.Module):
    def __init__(self, d_model, d_phi, hidden=128):
        super().__init__()
        self.k_ctx = nn.Linear(d_model, d_model, bias=False)
        self.k_tgt = nn.Linear(d_model, d_model, bias=False)
        self.k_phi = nn.Sequential(nn.Linear(d_phi, hidden), nn.ReLU(), nn.Linear(hidden, d_model))
        self.v_ctx = nn.Linear(d_model, d_model, bias=False)
        self.v_tgt = nn.Linear(d_model, d_model, bias=False)
        self.v_phi = nn.Sequential(nn.Linear(d_phi, hidden), nn.ReLU(), nn.Linear(hidden, d_model))
        # optional tiny gate
        self.gate = nn.Sequential(nn.Linear(3*d_model, d_model), nn.Sigmoid())

    def forward(self, Rc, Rt, dphi):
        # Rc: (B,Nt,Nc,D) broadcasted; Rt: same; dphi: (B,Nt,Nc,d_phi)
        K = self.k_ctx(Rc) + self.k_tgt(Rt) + self.k_phi(dphi)
        V = self.v_ctx(Rc) + self.v_tgt(Rt) + self.v_phi(dphi)
        g = self.gate(torch.cat([K, V, (K - V).abs()], dim=-1))
        return K * g, V * g

class TargetAwareContextAttention(nn.Module):
    """
    Cross-attention where K/V depend on (target j, context i) via pairwise features:
      pair_ji = [ R_ctx_i, R_tgt_j, dphi_ji ],  dphi_ji = phi_t[j] - phi_c[i]
    Shapes:
      R_t:   (B, Nt, D)
      R_ctx: (B, Nc, D)
      phi_t: (B, Nt, d_phi)
      phi_c: (B, Nc, d_phi)
      mask:  (B, Nc)  True=valid, False=pad
      value_weights: (B, Nc) optional weights per context row
    """
    def __init__(self, d_model, d_phi, n_heads=4, pair_hidden=128, out_dim=None, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q from targets as usual
        self.Wq = nn.Linear(d_model, d_model)

        # Build K,V from pairwise concat [R_ctx, R_t, dphi]
        #pair_in = 2 * d_model + d_phi
        #self.pair_mlp = MLP([pair_in] + pair_hidden + [2 * d_model])  # -> [K_raw | V_raw]
        self.factor_pair = FactorizedPairwiseKV(d_model, d_phi, pair_hidden)
        self.out = nn.Linear(d_model, out_dim or d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_q(self, x):
        # x: (B, Nt, D) -> (B, H, Nt, d_k)
        B, Nt, D = x.shape
        x = x.view(B, Nt, self.n_heads, self.d_k).transpose(1, 2)
        return x

    def _split_pair(self, x):
        # x: (B, Nt, Nc, D) -> (B, H, Nt, Nc, d_k)
        B, Nt, Nc, D = x.shape
        x = x.view(B, Nt, Nc, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        return x

    def forward(self, R_t, R_ctx, phi_t, phi_c, value_weights=None, mask=None, return_both_weights=False):      
        """
        If return_both_weights=True, value_weights must be given as a tuple (w_pos, w_neg),
        and we return (ctx_pos, ctx_neg). Otherwise return single ctx as before.
        """
        B, Nt, D = R_t.shape
        Nc = R_ctx.size(1)

        # 1) Pairwise features -> K_raw, V_raw  (compute once)
        Rt = R_t[:, :, None, :].expand(B, Nt, Nc, D)
        Rc = R_ctx[:, None, :, :].expand(B, Nt, Nc, D)
        dphi = phi_t[:, :, None, :] - phi_c[:, None, :, :]
        #pair = torch.cat([Rc, Rt, dphi], dim=-1)                       # (B,Nt,Nc,2D+d_phi)
        #kv = self.pair_mlp(pair.reshape(B * Nt * Nc, -1)).view(B, Nt, Nc, 2 * D)
        # K_raw, V_raw = kv.chunk(2, dim=-1)
        K_raw, V_raw = self.factor_pair(Rc, Rt, dphi)
        

        # 2) Heads + scores (once)
        Qh = self._split_q(self.Wq(R_t))                                # (B,H,Nt,d_k)
        Kh = self._split_pair(K_raw)                                    # (B,H,Nt,Nc,d_k)
        Vh_base = self._split_pair(V_raw)                               # (B,H,Nt,Nc,d_k)

        scores = torch.einsum('bhnd,bhncd->bhnc', Qh, Kh) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        def apply_weights(w):
            if w is None:
                Vh = Vh_base
            else:
                Vh = Vh_base * w[:, None, None, :, None]               # (B,1,1,Nc,1)
            ctx = torch.einsum('bhnc,bhncd->bhnd', attn, Vh)           # (B,H,Nt,d_k)
            ctx = ctx.transpose(1, 2).contiguous().view(B, Nt, D)
            return self.out(ctx)

        if return_both_weights:
            w_pos, w_neg = value_weights
            return apply_weights(w_pos), apply_weights(w_neg)
        else:
            return apply_weights(value_weights)
    
class TargetEncoder(nn.Module):
    """z_t = f([θ^t, φ^t, r^+_t, r^-_t])."""
    def __init__(self, theta_in_dim, phi_dim, r_dim, hidden, out_dim, theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim + 2*r_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, theta_t, phi_t, r_pos, r_neg):
        if r_pos.shape[1] == 1 and phi_t.shape[1] > 1:  # safety
            r_pos = r_pos.expand(-1, phi_t.shape[1], -1)
            r_neg = r_neg.expand_as(r_pos)
        theta_f = self.theta_enc(theta_t) if self.theta_enc else theta_t
        x = torch.cat([theta_f, phi_t, r_pos, r_neg], dim=-1)
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
class HCTargetAwareAttnNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=128):
        super().__init__()
        # Context side
        self.ctx_enc = ContextConditionalEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            theta_embed_dim=64, hidden=[128, 128], out_dim=d_model,
            mode='film', use_layernorm=True
        )

        # >>> NEW: context self-attention block(s) BEFORE pooling <<<
        self.ctx_sab = SAB(d_model=d_model, n_heads=4, ff_hidden=256, num_layers=2, dropout=0.1)

        # Target query R^{(t)}
        self.theta_enc_t = ThetaEncoder(d_theta, [128, 128], d_model)
        self.tquery = TargetQueryEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi,
            hidden=[128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )
        # >>> NEW target-aware attention <<<
        self.taware_attn = TargetAwareContextAttention(
            d_model=d_model, d_phi=d_phi, n_heads=4, pair_hidden=128, out_dim=d_model, dropout=0.0
        )

        # Attention pooling (two passes with different value weights)
        #self.attn = GlobalContextAttention(d_model, n_heads=4, out_dim=d_model)

        # Target encoder -> z_t
        self.tgt_enc = TargetEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi, r_dim=d_model,
            hidden=[128, 128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Bernoulli decoder
        self.decoder = BernoulliHead(4*d_model)

    def forward(self, theta_c, phi_c, y_c, theta_t, phi_t, mask_c=None):
        """
        theta_c: (B, Nc, d_theta)
        phi_c:   (B, Nc, d_phi)
        y_c:     (B, Nc, d_y)   -- assumed in [0,1] for weights
        theta_t: (B, Nt, d_theta)
        phi_t:   (B, Nt, d_phi)
        mask_c:  (B, Nc) optional boolean mask for valid context rows
        """
        B, Nc, _ = phi_c.shape
        _, Nt, _ = phi_t.shape

        # 1) Per-item context features
        R_ctx = self.ctx_enc(phi_c, y_c, theta_c)  # (B, Nc, D)

        # 1b) Let context items interact via SAB before pooling
        # nn.MultiheadAttention uses key_padding_mask=True for pads
        R_ctx = self.ctx_sab(R_ctx, key_padding_mask=(~mask_c.bool()) if mask_c is not None else None)  # (B, Nc, D)

        # 2) Build per-target query R^{(t)}
        R_t = self.tquery(theta_t, phi_t)                 # (B,Nt,D)

        # 3) Class target attention pooling (two value-weighted passes)
        wS = y_c.squeeze(-1).clamp(0, 1)                  # (B,Nc)
        wB = 1.0 - wS
        r_pos, r_neg = self.taware_attn(
            R_t, R_ctx, phi_t, phi_c, value_weights=(wS, wB), mask=mask_c, return_both_weights=True
        )
        #r_pos = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wS, mask=mask_c)  # (B,Nt,D)
        #r_neg = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wB, mask=mask_c)  # (B,Nt,D)
        
        # cosine similarity (B,Nt)
        cos = torch.nn.functional.cosine_similarity(r_pos, r_neg, dim=-1)
        #print("cos mean:", cos.mean().item(), "cos p95:", cos.quantile(0.95).item())
        # relative L2 gap (B,Nt)
        #rel = (r_pos - r_neg).norm(dim=-1) / (0.5*(r_pos.norm(dim=-1)+r_neg.norm(dim=-1))+1e-8)
        #print("rel mean:", rel.mean().item(), "rel p95:", rel.quantile(0.95).item())
        
        # 4) Target encoder -> z_t
        z_t = self.tgt_enc(theta_t, phi_t, r_pos, r_neg)  # (B,Nt,D)
        # ComparatorHead: prototype interactions for sharper separation
        feats = torch.cat([z_t, r_pos.expand_as(z_t), r_neg.expand_as(z_t), z_t*(r_pos-r_neg).expand_as(z_t)], dim=-1)
        # 5) Bernoulli decoder
        logit = self.decoder(feats)                         # (B,Nt,1)
        return logit, cos