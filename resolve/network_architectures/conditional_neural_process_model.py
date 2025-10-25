import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DeterministicEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, dropout_p=0.0):
        super(DeterministicEncoder, self).__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            print("encoder", prev, h)
            enc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout_p > 0:
                enc_layers.append(nn.Dropout(dropout_p))
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim)]
        print("encoder", prev, latent_dim)
        self.encoder = nn.Sequential(*enc_layers)

    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation.

        Args:
        context_x: Tensor of size of batches x observations x m_ch. For this regression
          task this corresponds to the x-values.
        context_y: Tensor of size bs x observations x d_ch. For this regression
          task this corresponds to the y-values.

        Returns:
            representation: The encoded representation averaged over all context 
            points.
        """

        # Concatenate x and y along the filter axes
        encoder_input = torch.cat((context_x, context_y), dim=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_context_points, _ = encoder_input.shape
        hidden = encoder_input.view(batch_size * num_context_points, -1)
        
        # Last layer without a ReLu
        hidden = self.encoder(hidden)
        # Bring back into original shape (# Flatten the output feature map into a 1D feature vector)
        hidden = hidden.view(batch_size, num_context_points, -1)

        # Aggregator: take the mean over all points
        representation = hidden.mean(dim=1)
        return representation
"""
def sigmoid_expectation(mu, sigma):
    # Bound the variance
    sigma = 0.1 + 0.9*torch.nn.functional.softplus(sigma)
    #sigma = 0.01 + 0.99*torch.nn.functional.softplus(sigma)
    
    y = torch.from_numpy(np.sqrt(1+3/np.pi**2*sigma.detach().numpy()**2))
    # Bound the divisor to > 0
    tmp0 = torch.where(y==0.,1e-4,0.)
    y=torch.add(y,tmp0)
    
    expectation = torch.sigmoid(mu/y) 
    var = expectation * (1-expectation) * (1-(1/y))
    #var = 0.01 + 0.99*torch.nn.functional.softplus(var)
    #tmp = torch.where(var==0.,1.e-4,0.)
    #var = torch.add(expectation, tmp)
    
    return expectation, var
"""
def sigmoid_expectation(mu, sigma):
    # Bound the variance
    sigma = 0.1 + 0.9 * F.softplus(sigma) #forces sigma into the range [0.1, ~1.0]
    #sigma = F.softplus(sigma) + 1e-6

    # Calculate y = sqrt(1 + 3 / pi^2 * sigma^2)
    y = torch.sqrt(1. + 3. / (np.pi ** 2) * sigma ** 2)

    # Prevent division by zero
    y = torch.clamp(y, min=1e-4)

    # Calculate expectation and variance
    expectation = torch.sigmoid(mu / y)
    var = expectation * (1. - expectation) * (1. - (1. / y))

    return expectation, var

class DeterministicDecoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        super(DeterministicDecoder, self).__init__()
        dec_layers = []
        prev = latent_dim
        for h in hidden_dims:
            print("decoder",prev, h)
            dec_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec_layers += [nn.Linear(prev, output_dim*2)]  # linear head for regression-like recon
        print("decoder",prev,output_dim*2)
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, representation, target_x, is_binary):
        """Decodes the individual targets.

        Args:
            representation: The encoded representation of the context
            target_x: The x locations for the target query

        Returns:
            dist: A multivariate Gaussian over the target points.
            mu: The mean of the multivariate Gaussian.
            sigma: The standard deviation of the multivariate Gaussian.   
        """

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, num_total_points, _ = target_x.shape
        representation = representation.unsqueeze(1).repeat([1, num_total_points, 1])

        # Concatenate the representation and the target_x
        input = torch.cat((representation, target_x), dim=-1)
        hidden = input.view(batch_size * num_total_points, -1)

        # Last layer without a ReLu
        hidden = self.decoder(hidden)

        # Bring back into original shape
        hidden = hidden.view(batch_size, num_total_points, -1)

        # Get the mean an the variance
        #mu, sigma = torch.split(hidden, 1, dim=-1)
        mu, sigma = torch.split(hidden, hidden.size(-1) // 2, dim=-1)
        #sigma = torch.ones_like(sigma) * 0.2 # fixing sigma for debugging

        # Map mu to a value between 0 and 1 and get the expectation and variance
        if is_binary==True:
            mu, sigma = sigmoid_expectation(mu, sigma)

        # Get the distribution
        # Ensure scale is strictly positive before passing to Normal distribution
        #sigma = torch.clamp(sigma, min=1e-4)  # OR
        sigma = F.softplus(sigma) + 1e-6

        return mu, sigma

class ConditionalNeuralProcess(nn.Module):
    def __init__(self, input_dim, representation_size, encoder_sizes, decoder_sizes, output_dim, dropout_p=0.0):
        super(ConditionalNeuralProcess, self).__init__()
        """Initialises the model
        """
        self._encoder = DeterministicEncoder(input_dim, representation_size, encoder_sizes, dropout_p)
    
        self._decoder = DeterministicDecoder(output_dim, representation_size+input_dim-output_dim, decoder_sizes)

    def forward(self, query_theta, query_phi, context_theta, context_phi, context_y, **kwargs):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                context_x: Array of shape batch_size x num_context x 1 contains the 
                    x values of the context points.
                context_y: Array of shape batch_size x num_context x 1 contains the 
                    y values of the context points.
                target_x: Array of shape batch_size x num_target x 1 contains the
                    x values of the target points.
            target_y: The ground truth y values of the target y. An array of 
                shape batchsize x num_targets x 1.

        Returns:
            log_p: The log_probability of the target_y given the predicted
            distribution.
            mu: The mean of the predicted distribution.
            sigma: The variance of the predicted distribution.
        """
        is_binary = kwargs.get("is_binary", True)
        
        context_x = torch.cat([context_theta, context_phi], dim=2) 
        target_x = torch.cat([query_theta, query_phi], dim=2) 

        # Pass query through the encoder and the decoder

        representation = self._encoder(context_x, context_y)
        mu, sigma = self._decoder(representation, target_x, is_binary)
        
        output = {
            "logits": [mu,sigma]
        }
        
        return output


