import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from itertools import combinations_with_replacement
from numpy.polynomial.legendre import Legendre
import random
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from resolve.utilities import ModelVisualizer
from scipy.integrate import nquad
from math import comb
import pymc as pm
# Set seeds for reproducibility
np.random.seed(42)         # NumPy seed
random.seed(42)            # Python random seed

class PCEMultiFidelityModelVisualizer(ModelVisualizer):
    def __init__(self, fidelities, parameters, degree, trace=None):
        """
        Initialize the multi-fidelity model visualizer.
        Parameters:
        - basis_matrices (dict): Dictionary of basis matrices for each fidelity level.
          Example: {"lf": basis_matrix_lf, "mf": basis_matrix_mf, "hf": basis_matrix_hf}
        - indices (dict): Dictionary of indices mapping one fidelity level to the next.
          Example: {"mf": indices_mf, "hf": indices_hf}
        - priors (dict): Dictionary of prior configurations for each fidelity level.
          Example: {"lf": {"sigma": 0.5}, "mf": {"sigma": 0.1}, "hf": {"sigma": 0.01}}
        """
        super().__init__(fidelities, parameters)
        self.degree = degree
        self.trace = trace
        if trace==None:
            print("Warring: No trace has been given. Please run \"read_trace(path_to_trace)\"")
        self.y_scaling =  [1.0]*self.nfidelities
        self.y_marginalized = None
        #self.get_marginalized()

    def read_trace(self, path_to_trace,version="v1.0"):
        self.trace = az.from_netcdf(f"{path_to_trace}/pce_{version}_trace.nc")

    def normalize_to_minus1_plus1(self,x):
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
    
    def reverse_normalize(self, x_norm):
        return self.x_min + (x_norm + 1) * (self.x_max - self.x_min) / 2
  
    def _generate_basis(self, x_data, degree):
        """
        Generate the multivariate Legendre basis for multi-dimensional inputs.

        Parameters:
        - x_data (ndarray): Input data of shape (n_samples, n_dim).

        Returns:
        - basis_matrix (ndarray): Shape (n_samples, n_terms).
        """
        n_samples, n_dim = x_data.shape
        degree_new = degree
        
        terms = []
        # Generate all combinations of terms up to the given degree
        for deg in range(degree_new + 1):
            for combo in combinations_with_replacement(range(n_dim), deg):
                terms.append(combo)

        # Evaluate each term for all samples
        basis_matrix = np.zeros((n_samples, len(terms)))
        for i, term in enumerate(terms):
            poly = np.prod([Legendre.basis(1)(x_data[:, dim]) for dim in term], axis=0)
            basis_matrix[:, i] = poly
        return basis_matrix, degree_new
    
    def generate_y_pred_samples(self, x_data, include_noise=False):
        """
        Generate high-fidelity prediction samples based on posterior trace.
        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_pred_samples (ndarray): A lit of each predicted fidelity samples (shape: list of (n_samples_total x n_hf_samples)).
        """
        y_pred_samples=[]

        basis_matrix_test,_ = self._generate_basis(x_data,self.degree[0])  # Shape: (n_samples, n_terms_hf)
        coeff_samples = self.trace.posterior[f"coeffs_{self.fidelities[0]}"].values
        coeff_samples_flat = coeff_samples.reshape(-1, coeff_samples.shape[-1]) 
        y_pred = np.dot(coeff_samples_flat, basis_matrix_test.T)
        y_pred = np.exp(y_pred)


        if include_noise:
            sigma = self.trace.posterior[f"sigma_{self.fidelities[0]}"].values.flatten()
            noise = np.random.normal(0, sigma[:, None], size=y_pred.shape)
            y_pred += noise

        y_pred_samples.append(y_pred)  # Shape: (n_samples_total, n_lf_samples)

        for i,f in enumerate(self.fidelities[1:]):
            # Extract coefficients from the posterior
            coeff_samples_delta = self.trace.posterior[f"coeffs_delta_{f}"].values  # Shape: (n_chains, n_draws, n_terms_hf)
            coeff_samples_delta_flat = coeff_samples_delta.reshape(-1, coeff_samples_delta.shape[-1])  # Shape: (n_samples_total, n_terms_hf)
            basis_matrix_test,_ = self._generate_basis(x_data,self.degree[i+1])
            delta_pred_samples = np.dot(coeff_samples_delta_flat, basis_matrix_test.T)  # Shape: (n_samples_total, n_hf_samples)
            rho_samples = self.trace.posterior[f"rho_{f}"].values  # Shape: (n_chains, n_draws)
            rho_samples_flat = rho_samples.flatten()  # Shape: (n_samples_total,)
            y_pred = rho_samples_flat[:, None] * y_pred_samples[-1] + delta_pred_samples
            y_pred = np.exp(y_pred)

            if include_noise:
                sigma = self.trace.posterior[f"sigma_{f}"].values.flatten()
                noise = np.random.normal(0, sigma[:, None], size=y_pred.shape)
                y_pred += noise

            # Compute HF predictions
            y_pred_samples.append(y_pred)  # Shape: (n_samples_total, n_hf_samples)
        return y_pred_samples

    def predict(self, x_data, fidelity=1):
        """
        Predict the high-fidelity output for a given input using the model.
        Parameters:
        - x_data (ndarray): Input data (e.g., validation or test set).
        - trace: Trace object containing posterior samples from PyMC.

        Returns:
        - y_pred_samples (ndarray): A list of each predicted fidelity samples (shape: list of (n_samples_total x n_hf_samples)).
        """
        x_data = self.normalize_to_minus1_plus1(x_data)
        y_pred_samples = self.generate_y_pred_samples(x_data)[fidelity]
        return y_pred_samples
    
    def get_model_prediction(self, x_data):
        
        y_mean=[]
        sigma1=[]
        sigma2=[]
        sigma3=[]
        for f in range(self.nfidelities):
            x_data_tmp = self.normalize_to_minus1_plus1(x_data[f])
            y_pred_samples = self.generate_y_pred_samples(x_data_tmp)[f]
            y_mean.append(np.percentile(y_pred_samples, 50., axis=0))
            sigma1.append([np.percentile(y_pred_samples, 16., axis=0),np.percentile(y_pred_samples, 84., axis=0)])
            sigma2.append([np.percentile(y_pred_samples, 2.5, axis=0),np.percentile(y_pred_samples, 97.5, axis=0)])
            sigma3.append([np.percentile(y_pred_samples, 0.5, axis=0),np.percentile(y_pred_samples, 99.5, axis=0)])

        return y_mean, sigma1, sigma2, sigma3

    def unnormalized_pdf(self, x, fidelity=1):
        pred = self.predict(x, fidelity)
        return np.maximum(pred, 0)

    def normalized_pdf(self, x, bounds, fidelity):
        norm_factor, _ = nquad(self.unnormalized_pdf, bounds)
        return self.unnormalized_pdf(x) / norm_factor
    
    def likelihood(self, x, fidelity=1):
        pred = self.predict(x, fidelity)
        return np.maximum(pred, 1e-12)  # avoid log(0)

    def log_likelihood(self, x):
        return np.log(self.likelihood(x))

    def get_marginalized(self, grid_steps=20):
            def reverse_norm(x_norm, x_min, x_max):
                return x_min + (x_norm + 1) * (x_max - x_min) / 2

            x_grid_norm_list = []
            self.x_grid = []
            grid_steps_list = []
            for i in range(len(self.x_min)):
                grid_steps_list.append(grid_steps)
                arr = np.linspace(-1., 1., grid_steps)
                x_grid_norm_list.append(arr)
                self.x_grid.append(reverse_norm(x_grid_norm_list[-1],self.x_min[i],self.x_max[i]))

            mesh = np.meshgrid(*x_grid_norm_list, indexing='ij')
            x_grid = np.column_stack([x.flatten() for x in mesh])  # shape: (m, 4)
            y = self.generate_y_pred_samples(x_grid) # is a list of shape (n_posterior_draws, n_data_samples)

            self.y_marginalized = []
            for f in range(self.nfidelities):
                self.y_marginalized.append([])
                
                for ix in range(len(self.x_min)):
                    y_grid = y[f].reshape(len(y[f]),*grid_steps_list)
                    # Define axes to marginalize (all axes except the kept one)
                    all_axes = list(range(1, y_grid.ndim))  # skip the draw axis (axis 0)
                    marg_axes = tuple(ax for ax in all_axes if ax != (ix + 1))
                    self.y_marginalized[-1].append(np.mean(y_grid, axis=marg_axes))

    def plot_marginalized(self,grid_steps=50):
        
        self.get_marginalized(grid_steps=grid_steps)
        nrows = self.nfidelities
        ncols = len(self.x_max)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),squeeze=False)

        for f in range(self.nfidelities):
            for keep_axis in range(len(self.x_max)):
                ax = axes[f][keep_axis]

                y_mean = np.percentile(self.y_marginalized[f][keep_axis], 50., axis=0)
                y_1sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 16., axis=0)
                y_1sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 84., axis=0)
                y_2sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 2.5, axis=0)
                y_2sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 97.5, axis=0)
                y_3sigma_low = np.percentile(self.y_marginalized[f][keep_axis], 0.5, axis=0)
                y_3sigma_high = np.percentile(self.y_marginalized[f][keep_axis], 99.5, axis=0)

                ax.fill_between(
                    self.x_grid[keep_axis], y_3sigma_low, y_3sigma_high,
                    color="coral", alpha=0.2, label=r'$\pm 3\sigma$'
                )
                ax.fill_between(
                    self.x_grid[keep_axis], y_2sigma_low, y_2sigma_high,
                    color="yellow", alpha=0.2, label=r'$\pm 2\sigma$'
                )
                ax.fill_between(
                    self.x_grid[keep_axis], y_1sigma_low, y_1sigma_high,
                    color="green", alpha=0.2, label=r'$\pm 1\sigma$'
                )

                #y_ax=np.nan_to_num(y, nan=0.0)

                ax.set_xlabel(f'{self.feature_labels[keep_axis]}',fontsize=16)
                str_tmp = f"{self.fidelities[f]}"
                ax.set_ylabel('Marginalized predicted $\epsilon^{('+str_tmp+')}$', fontsize=16)


        n_rows = axes.shape[0]
        for i in range(n_rows - 1):
            for ax in axes[i, :]:
                ax.set_xlabel("")
        
        #for i in range(n_rows):
        #    for ax in axes[i, 1:]:
        #        ax.set_ylabel("")

        # Create custom legend handles

        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Model prediction'),
            mpatches.Patch(color='green', alpha=0.2, label=r'$\pm 1\sigma$'),
            mpatches.Patch(color='yellow', alpha=0.2, label=r'$\pm 2\sigma$'),
            mpatches.Patch(color='coral', alpha=0.2, label=r'$\pm 3\sigma$')
        ]

        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), fontsize=16, frameon=False, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space at bottom for legend
        return fig

    def get_marginalized_random(self, x_data, keep_axis=0, grid_steps=10):

        for f in self.fidelities:
            x_data_normalized = self.normalize_to_minus1_plus1(x_data)
        
        #    y_hf is assumed to have shape (n_posterior_draws, n_samples)
        y_hf = self.y_scaling[-1] * self.generate_y_pred_samples(x_data_normalized)[-1]
        
        # Extract the keep_axis values from the random inputs and reverse the normalization.
        x_keep = x_data[:, keep_axis]

        # Define bins along the kept axis (using the unnormalized space for plotting).
        bin_edges = np.linspace(self.x_min[keep_axis], self.x_max[keep_axis], grid_steps + 1)
        # Compute the bin centers for plotting.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # For each bin, compute the average predicted y for each posterior draw.
        n_draws = y_hf.shape[0]
        binned_means = np.empty((n_draws, grid_steps))
        
        # Loop over each bin to compute means.
        for i in range(grid_steps):
            # Define a mask for the samples falling in the current bin.
            # For all but the last bin, use a half-open interval; include the right edge only for the last bin.
            if i < grid_steps - 1:
                bin_mask = (x_keep >= bin_edges[i]) & (x_keep < bin_edges[i+1])
            else:
                bin_mask = (x_keep >= bin_edges[i]) & (x_keep <= bin_edges[i+1])
            
            # Check if there are any samples in the bin.
            if np.sum(bin_mask) == 0:
                # If no samples fall in the bin, assign a NaN value.
                binned_means[:, i] = np.nan
            else:
                # For each posterior draw, average the predictions of the samples in the bin.
                # y_hf has shape (n_draws, n_samples), so for each draw we average over the masked indices.
                binned_means[:, i] = np.mean(y_hf[:, bin_mask], axis=1)
        
        # Compute the percentiles across posterior draws for each bin.
        y_hf_mean       = np.nanpercentile(binned_means, 50, axis=0)
        y_hf_1sigma_low = np.nanpercentile(binned_means, 16, axis=0)
        y_hf_1sigma_high= np.nanpercentile(binned_means, 84, axis=0)

        return y_hf_mean, y_hf_1sigma_low, y_hf_1sigma_high