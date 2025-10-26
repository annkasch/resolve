import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE):
    x -> encoder -> (mu, logvar) -> sample z -> decoder -> x_hat
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64], dropout_p=0.0):
        super().__init__()

        # --- Encoder ---
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                enc_layers.append(nn.Dropout(dropout_p))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Separate linear heads for mean and log-variance
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # --- Decoder ---
        dec_layers = []
        rev_hidden = list(hidden_dims)[::-1]
        prev = latent_dim
        for h in rev_hidden:
            dec_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]  # Linear output layer for regression-like reconstruction
        self.decoder = nn.Sequential(*dec_layers)

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, query_theta, query_phi, **kwargs):
        # Combine theta and phi (same interface as AE)
        x = torch.cat([query_theta, query_phi], dim=2)

        # Encode to latent distribution
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Sample z
        z = self.reparameterize(mu, logvar)

        # Decode to reconstruct input
        x_hat = self.decoder(z)

        # KL Divergence term (for loss computation)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        output = {
            "logits": [x_hat],
            "mu": mu,
            "logvar": logvar,
            "kl_div": kl_div
        }
        return output

    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @torch.no_grad()
    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def reconstruct(self, query_theta, query_phi):
        return self.forward(query_theta, query_phi)