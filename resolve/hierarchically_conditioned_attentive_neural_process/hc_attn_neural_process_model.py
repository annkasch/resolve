import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Utility ========== #
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


# ========== Encoders ========== #
class ThetaEncoder(nn.Module):
    """Encodes θ into a fixed-size embedding."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, theta):
        leading = theta.shape[:-1]     # (B, N?) possibly missing N
        flat = theta.reshape(-1, theta.shape[-1])
        z = self.mlp(flat)
        return z.view(*leading, -1)


class ContextConditionalEncoder(nn.Module):
    """Encodes (φ^c, y^c) conditioned on θ^c."""
    def __init__(self, phi_dim, y_dim, theta_in_dim,
                 theta_embed_dim, hidden, out_dim, mode='concat'):
        super().__init__()
        self.mode = mode
        self.theta_enc = ThetaEncoder(theta_in_dim, [], theta_embed_dim)

        if mode == 'concat':
            in_dim = phi_dim + y_dim + theta_embed_dim
            self.content = MLP([in_dim] + hidden + [out_dim])
        elif mode == 'film':
            dims = [phi_dim + y_dim] + hidden
            self.feature_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.feature_layers.append(nn.Linear(dims[i], dims[i+1]))
                self.film_layers.append(nn.Linear(theta_embed_dim, 2 * dims[i+1]))
            self.final = nn.Linear(hidden[-1], out_dim)
        else:
            raise ValueError("mode must be 'concat' or 'film'")

    def forward(self, phi_c, y_c, theta_c):
        B, Nc, _ = phi_c.shape
        theta_emb = self.theta_enc(theta_c)  # (B, Nc, Eθ)

        if self.mode == 'concat':
            inp = torch.cat([phi_c, y_c, theta_emb], dim=-1)
            h = self.content(inp.view(B*Nc, -1))
            return h.view(B, Nc, -1)

        # FiLM mode
        h = torch.cat([phi_c, y_c], dim=-1).view(B*Nc, -1)
        cond = theta_emb.view(B*Nc, -1)
        for lin, film in zip(self.feature_layers, self.film_layers):
            h = lin(h)
            gamma, beta = film(cond).chunk(2, dim=-1)
            h = F.relu(gamma * h + beta)
        h = self.final(h)
        return h.view(B, Nc, -1)


class GlobalContextAttention(nn.Module):
    """Multi-head attention pooling to compute r_θ (supports separate K/V and value weights)."""
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

    def forward(self, R_theta, K_src, V_src=None, value_weights=None, mask=None):
        """
        R_theta:      (B, Nq, D)
        K_src:        (B, Nc, D)   # keys come from unweighted context
        V_src:        (B, Nc, D)   # values (defaults to K_src if None)
        value_weights:(B, Nc)      # optional per-position weights for V
        mask:         (B, Nc)      # optional valid mask for keys/values
        """
        if V_src is None: V_src = K_src

        Q = self._split(self.Wq(R_theta))              # (B,H,Nq,d_k)
        K = self._split(self.Wk(K_src))                # (B,H,Nc,d_k)
        V = self._split(self.Wv(V_src))                # (B,H,Nc,d_k)

        if value_weights is not None:
            # broadcast weights onto heads and d_k
            w = value_weights[:, None, :, None]  # (B,1,1,Nc,1)
            V = V * w

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,H,Nq,Nc)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)                # (B,H,Nq,d_k)

        B, H, Nq, d_k = context.shape
        context = context.transpose(1, 2).contiguous().view(B, Nq, H * d_k)
        return self.out(context)                       # (B,Nq,D_out)


class TargetEncoder(nn.Module):
    """Fuses θ^t, φ^t, and r_θ into target embeddings."""
    def __init__(self, theta_in_dim, phi_dim, r_dim, hidden, out_dim,
                 theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim + r_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, theta_t, phi_t, r_theta):
        B, Nt, _ = phi_t.shape
        theta_f = self.theta_enc(theta_t) if self.theta_enc else theta_t
        if r_theta.shape[1] == 1 and Nt > 1:
            r_theta = r_theta.expand(-1, Nt, -1)
        x = torch.cat([theta_f, phi_t, r_theta], dim=-1)
        h = self.mlp(x.view(B*Nt, -1))
        return h.view(B, Nt, -1)


# ========== Decoder Head ========== #
class DecoderHead(nn.Module):
    """Predicts Gaussian parameters (μ, log σ) for y^t."""
    def __init__(self, in_dim, y_dim, hidden=[128, 128]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [2*y_dim])

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        out = self.net(z_t.view(B*Nt, -1))
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu.view(B, Nt, -1), log_sigma.view(B, Nt, -1)

class BernoulliHead(nn.Module):
    """Maps target embedding to a logit for P(signal)."""
    def __init__(self, in_dim, hidden=[256, 256]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [1])  # 1 logit

    def forward(self, z_t):  # z_t: (B, Nt, in_dim)
        B, Nt, _ = z_t.shape
        logit = self.net(z_t.view(B*Nt, -1)).view(B, Nt, 1)
        return logit
    
# ========== Full Model ========== #
class HCContextAttnNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=128):
        super().__init__()
        # Context encoders
        self.theta_enc_c = ThetaEncoder(d_theta, [128, 128], d_model)
        
        self.ctx_enc = ContextConditionalEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            theta_embed_dim=64, hidden=[128, 128], out_dim=d_model,
            mode='film'
        )
        
        self.attn = GlobalContextAttention(d_model, n_heads=4, out_dim=d_model)
        
        # Target encoder
        self.theta_enc_t = ThetaEncoder(d_theta, [128, 128], d_model)
        self.tgt_enc = TargetEncoder(d_theta, d_phi, r_dim=2*d_model,
                                     hidden=[128, 128], out_dim=d_model,
                                     theta_encoder=self.theta_enc_t)

        # Final decoder head
        #self.decoder = DecoderHead(d_model, d_y)
        self.decoder = BernoulliHead(4*d_model)

    def forward(self, theta_c, phi_c, y_c, theta_t, phi_t):
        """
        Args:
            theta_c: (B, Nc, d_theta)
            phi_c:   (B, Nc, d_phi)
            y_c:     (B, Nc, d_y)
            theta_t: (B, Nt, d_theta)
            phi_t:   (B, Nt, d_phi)
        Returns:
            mu, log_sigma for targets (B, Nt, d_y)
        """

        # Encoded context once
        R_c_theta = self.ctx_enc(phi_c, y_c, theta_c)           # (B,Nc,D)
        R_theta   = self.theta_enc_c(theta_c.mean(1, True))     # (B,1,D)

        wS = y_c.squeeze(-1).clamp(0,1)                    # (B,Nc)
        wB = 1.0 - wS
        print(wS.shape, wB.shape, y_c.shape, R_c_theta.shape, R_theta.shape)
        # Same keys, different (weighted) values:
        r_pos = self.attn(R_theta, K_src=R_c_theta, V_src=R_c_theta, value_weights=wS)  # (B,1,D)
        r_neg = self.attn(R_theta, K_src=R_c_theta, V_src=R_c_theta, value_weights=wB)  # (B,1,D)
        cos = torch.nn.functional.cosine_similarity(r_pos, r_neg, dim=-1)

        print(r_pos.shape, r_neg.shape, cos.shape)

        r_both = torch.cat([r_pos, r_neg], dim=-1)         # (B,1,2D)

        z_t = self.tgt_enc(theta_t, phi_t, r_both)

        #mu, log_sigma = self.decoder(z_t)
        #return mu, log_sigma
        # ComparatorHead: prototype interactions for sharper separation
        feats = torch.cat([z_t, r_pos.expand_as(z_t), r_neg.expand_as(z_t), z_t*(r_pos-r_neg).expand_as(z_t)], dim=-1)
        logit = self.decoder(feats)
        return logit,cos
        
