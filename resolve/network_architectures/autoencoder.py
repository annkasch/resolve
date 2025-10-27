import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """
    Standard AE: x -> encoder -> z -> decoder -> x_hat
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[128, 64], dropout_p=0.0):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                enc_layers.append(nn.Dropout(dropout_p))
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        rev_hidden = list(hidden_dims)[::-1]
        prev = latent_dim
        for h in rev_hidden:
            dec_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]  # linear head for regression-like recon
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, query_theta, query_phi, **kwargs):
        target_x = torch.cat([query_theta, query_phi], dim=2) 
        z = self.encoder(target_x)
        x_hat = self.decoder(z)
        output = {
            "logits": [x_hat]
        }
        return output

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)