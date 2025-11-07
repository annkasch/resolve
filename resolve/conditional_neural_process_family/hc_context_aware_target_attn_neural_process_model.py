import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.class_attention import GlobalContextAttentionDual,  GlobalContextAttention
from resolve.conditional_neural_process_family.memory_bank import MemoryBank
from resolve.conditional_neural_process_family.target_encoder import TargetQueryEncoder, TargetEncoder
from resolve.conditional_neural_process_family.context_encoder import ContextConditionalEncoder, ThetaEncoder, MLP
from resolve.conditional_neural_process_family.transformer_encoder import ContextTransformerEncoder

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
    def __init__(self, d_theta, d_phi, d_y, d_model=32, encoder_sizes=[128,128], theta_embed_dim=32, n_heads=4, mode = 'film'):
        super().__init__()
        # Context side
        #self.ctx_enc2 = ContextConditionalEncoder(
        #    phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
        #    theta_embed_dim=theta_embed_dim, hidden=encoder_sizes, out_dim=d_model,
        #    mode=mode, use_layernorm=True
        #)
        self.ctx_enc = ContextTransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi,
            y_dim=d_y,
            embed_dim= d_model,
            depth= 4,
            num_heads= 4,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= False       # set False: mean over feature tokens
        )

        # Memory bank
        self.mem_key = nn.Linear(d_model + theta_embed_dim, d_model, bias=False)   # for writes
        self.mem_qry = nn.Linear(d_model + theta_embed_dim, d_model, bias=False)   # for reads
        self.theta_key_enc = ThetaEncoder(d_theta, [], theta_embed_dim)            # small θ emb for memory keys/queries

        # memory config
        num_theta_cells = 64          # choose based on your θ coverage (or pass in)
        self.memory = MemoryBank(
            num_theta_cells=1,   # POC: single bucket
            dim=d_model,
            J_pos=5000, J_neg=50000,
            alpha=0.95, tau_assign=0.45
        )

        self.theta_enc_t = ThetaEncoder(d_theta, encoder_sizes, d_model)
        
        # Target query R^{(t)}
        self.tquery = TargetQueryEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi,
            hidden=encoder_sizes, out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Attention pooling (two passes with different value weights)
        #self.attn = GlobalContextAttention(d_model, n_heads=4, out_dim=d_model)
        self.attn = GlobalContextAttentionDual(d_model, n_heads=n_heads, out_dim=d_model)
        
        # Target encoder -> z_t
        self.tgt_enc = TargetEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi, r_dim=d_model,
            hidden=encoder_sizes, out_dim=d_model, theta_encoder=self.theta_enc_t
        )
        
        # Bernoulli decoder
        self.decoder = BernoulliHead(d_model)


    def forward(self, query_theta, query_phi, context_theta, context_phi, context_y, qry_theta_cell=None, return_ctx_for_write=True, **kwargs):
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
        #R_ctx2 = self.ctx_enc2(context_phi, context_y, context_theta) 
        # 2) Build per-target query R^{(t)}
        R_t = self.tquery(query_theta, query_phi)                 # (B,Nt,D)

        # --- build memory query q = norm( MLP([R_t, Eθ]) ) ---
        theta_small = self.theta_key_enc(query_theta)                # (B,Nt,Eθm)
        q_mem = torch.cat([R_t, theta_small], dim=-1)                # (B,Nt,D+Eθm)
        q_mem = F.normalize(self.mem_qry(q_mem), dim=-1)             # (B,Nt,D)

        self.mem_gate = nn.Parameter(torch.tensor(0.5))  # starts at 0.5, learnable

        # guard: if no cell ids provided, skip memory
        # --- Memory read (robust) ---
        # 3) Class target attention pooling (two value-weighted passes)

        wS = (context_y.squeeze(-1) > 0.5)   # bool mask for positives
        wB = ~wS                           # inverse mask
        
        #R_ctx_neg = R_ctx[wB].unsqueeze(0).repeat(B, 1, 1)
        """
        if qry_theta_cell is not None:
            dev = R_t.device
            B, Nt, D = R_t.shape

            if qry_theta_cell.dtype != torch.long:
                qry_theta_cell = qry_theta_cell.long()
            if qry_theta_cell.shape != (B, Nt):
                qry_theta_cell = qry_theta_cell.view(B, -1).expand(B, Nt).contiguous()
            qry_theta_cell = qry_theta_cell.to(dev)

            # cap K to capacity; start tiny for POC
            Kp = min(3, int(self.memory.pos.size(1)))   # J_pos can be 3–5; use 4 neighbors

            if self.memory.pos_mask.any():
                r_mem, r_pos, r_neg = self.memory.read(q=q_mem, qry_theta_cell=qry_theta_cell,
                                            K_pos=5, K_neg=5, tau=0.1, lambda_neg=0.5)
                R_ctx_pos=r_pos
                R_ctx_neg=r_neg


            else:
                R_ctx_pos = R_ctx[wS].unsqueeze(0).repeat(B, 1, 1)
                R_ctx_neg = R_ctx[wB].unsqueeze(0).repeat(B, 1, 1)
        """
        # single attention pass -> both r_pos and r_neg
        r_pos, r_neg = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, wS=wS, mask=mask_c)
        #r_pos2, r_neg2 = self.attn(Q_src=R_t, K_src=R_ctx2, V_src=R_ctx2, wS=wS, mask=mask_c)
        #print(r_pos.shape, r_neg.shape)

        #r_pos = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wS)  # (B,Nt,D)
        #r_neg = self.attn(Q_src=R_t, K_src=R_ctx, V_src=R_ctx, value_weights=wB)  # (B,Nt,D)
        #print("att mean 1:", r_pos.mean().item(), r_neg.mean().item())
        
        #cos = torch.nn.functional.cosine_similarity(r_pos, r_neg, dim=-1)
        #print("cos mean1:", cos.mean().item(), "cos p95:", cos.quantile(0.95).item())

        #r_pos, r_neg, loss = self.attn(Q_src = R_t, K_src_pos=R_ctx_pos, K_src_neg=R_ctx_neg, V_src_pos=R_ctx_pos, V_src_neg=R_ctx_neg)  # (B,Nt,D)
        #r_neg = self.attn(Q_src=R_t, K_src=R_ctx_neg, V_src=R_ctx_neg)  # (B,Nt,D)
        #print("att mean 2:", r_pos.mean().item(), r_neg.mean().item())

        
        
        # 5) ComparatorHead: prototype interactions for sharper separation
        #feats = torch.cat([z_t, r_pos.expand_as(z_t), r_neg.expand_as(z_t), z_t*(r_pos-r_neg).expand_as(z_t)], dim=-1)
        # cosine similarity (B,Nt)

        # Contrast and interaction features
        diff = r_pos - r_neg
        m_pos = R_t * r_pos
        m_neg = R_t * r_neg
        cos = torch.nn.functional.cosine_similarity(r_pos, r_neg, dim=-1)
        print("cos mean:", cos.mean().item(), "cos p95:", cos.quantile(0.95).item())
        #cos2 = torch.nn.functional.cosine_similarity(r_pos2, r_neg2, dim=-1)
        #print("cos mean2:", cos2.mean().item(), "cos p95:", cos2.quantile(0.95).item())
        # relative L2 gap (B,Nt)
        #rel = (r_pos - r_neg).norm(dim=-1) / (0.5*(r_pos.norm(dim=-1)+r_neg.norm(dim=-1))+1e-8)
        #print("rel mean:", rel.mean().item(), "rel p95:", rel.quantile(0.95).item())
        
        # 4) Target encoder -> z_t
        z_t = self.tgt_enc(query_theta, query_phi, r_pos, r_neg, diff, m_pos, m_neg)  # (B,Nt,D)

        # 6) Bernoulli decoder
        logit = self.decoder(z_t)                         # (B,Nt,1)

        output = {
            "logits": [logit],
            "cosine_sim": cos,
        #    "loss": loss
        }
        if return_ctx_for_write:
            output["R_ctx_for_write"] = R_ctx.detach()  # no graph, no grads
        
        return output
    
    @torch.no_grad()
    def build_mem_keys(self, context_theta, R_ctx):
        # small θ embedding already defined in __init__: self.theta_key_enc
        theta_small = self.theta_key_enc(context_theta)              # (B,Nc,Eθm)
        k = torch.cat([R_ctx, theta_small], dim=-1)                  # (B,Nc,D+Eθm)
        k = F.normalize(self.mem_key(k), dim=-1)                     # (B,Nc,D)
        return k