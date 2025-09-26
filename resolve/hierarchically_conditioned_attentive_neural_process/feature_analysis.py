
import umap
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import umap
import numpy as np

class UMAPAnalyzer:
    def __init__(self, batch_size=64, device="cpu", n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean"):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to analyze.
        model : torch.nn.Module, optional
            Model to embed data before UMAP (e.g., a feature extractor). If None, raw data is used.
        batch_size : int
            Batch size for DataLoader.
        device : str
            'cpu' or 'cuda'.
        n_neighbors, min_dist, n_components, metric : UMAP parameters
        """
        self.batch_size = batch_size
        self.umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "metric": metric
        }
        self.embeddings = None
        self.reducer = None

    def extract_features(self, dataset):
        """Run the dataset through the model (if provided) and return embeddings."""
        loader = dataset.dataloader
        theta_dim = len(dataset._names_theta)
        phi_dim = len(dataset._names_phi)
        y_dim = len(dataset._names_target)
        features_theta = []
        features_phi = []

        with torch.no_grad():
            for batch in loader:

                x_theta = batch[:, 0:theta_dim]
                x_phi = batch[:, theta_dim:theta_dim + phi_dim]
                y_label = batch[:, theta_dim + phi_dim:theta_dim + phi_dim + y_dim]
                x_theta = x_theta.to(self.device, dtype=torch.float)
                x_phi = x_phi.to(self.device, dtype=torch.float)
                y_label = y_label.to(self.device, dtype=torch.float)

                features_theta.append((x_theta.cpy().numpy()))
                features_phi.append(x_phi.cpy().numpy())

        self.embeddings_theta = np.concatenate(features_theta, axis=0)
        self.embeddings_phi = np.concatenate(features_phi, axis=0)
    
    def add_features(x):

    def fit_umap(self):
        """Run UMAP on extracted embeddings."""
        if self.embeddings_phi is None:
            self.extract_features()

        self.reducer = umap.UMAP(**self.umap_params)
        self.reduced = self.reducer.fit_transform(self.embeddings)
        return self.reduced

    def get_results(self):
        """Return the low-dimensional representation."""
        if self.reduced is None:
            self.fit_umap()
        return self.reduced