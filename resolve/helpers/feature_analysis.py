
from cProfile import label
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt 
import umap
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import itertools
import torch
from scipy.ndimage import gaussian_filter

class UMAPAnalyzer:
    def __init__(self, batches, device="cpu", n_neighbors=30, min_dist=0.1, n_components=2, metric='cosine', n_epochs=120, negative_sample_rate=5, low_memory=True, random_state=42, verbose=True):
        self.batches = batches
        self.umap_params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "metric": metric,
            "n_epochs": n_epochs,
            "random_state": random_state,
            "low_memory": low_memory,
            "verbose": verbose,
            "negative_sample_rate": negative_sample_rate,
        }

        self.embeddings = {}
        self.labels = None
        self.reducer = None

    def extract_features(self, loader):
        """Run the dataset through the model (if provided) and return embeddings."""

        features_theta = []
        features_phi = []
        y_labels = []

        #with torch.no_grad():
        for i, batch in enumerate(loader):
            if i not in self.batches:
                continue
            _, query, target = batch
            
            features_theta.append(query.theta[0])
            features_phi.append(query.phi[0])
            y_labels.append(target[0])
            

        self.embeddings = {
            "theta": np.concatenate(features_theta, axis=0),
            "phi": np.concatenate(features_phi, axis=0)
        }
        self.labels = np.concatenate(y_labels, axis=0)
    

    def fit_umap(self, embeddings="global"):
        """Run UMAP on extracted embeddings."""
        if self.embeddings is {}:
            self.extract_features()

        self.reducer = umap.UMAP(**self.umap_params)

        if embeddings == "global":
            X =  np.hstack([self.embeddings["theta"], self.embeddings["phi"]])
        else:
            X = self.embeddings["phi"]
        
        self.reduced = self.reducer.fit_transform(X)
        return self.reduced

    def get_results(self):
        """Return the low-dimensional representation."""
        if self.reduced is None:
            self.fit_umap()
        return self.reduced
    
    def plot(self, mode=""):
        plt.figure(figsize=(8,6))
        if mode == "density":
            plt.figure(figsize=(8, 6))
            plt.hexbin(
                self.reduced[:, 0], self.reduced[:, 1],
                gridsize=200, cmap='viridis'
            )
            plt.colorbar(label='Density')
            plt.title("UMAP density projection")
        else:
            plt.scatter(
                self.reduced[:, 0],
                self.reduced[:, 1],
                c=self.labels,
                cmap='cool',
                s=3,
                alpha=0.5
            )
            plt.title("UMAP projection of signal vs background")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
    
    def log_signal_over_background_kde(self, embeddings="global"):
        if self.embeddings is {}:
            self.extract_features()
        if self.reducer is None:
            self.fit_umap(embeddings=embeddings)

        if embeddings == "global":
            E =  np.hstack([self.embeddings["theta"], self.embeddings["phi"]])
        else:
            E = self.embeddings["phi"]
        E = np.asarray(E)   # your UMAP coords
        Y = np.asarray(self.labels).ravel()
        # --- masks (row-wise indexing) ---
        ms = (Y == 1)   # signal
        mb = (Y == 0)   # background

        grid_x, grid_y = np.mgrid[
            np.min(E[:,0]):np.max(E[:,0]):200j,
            np.min(E[:,1]):np.max(E[:,1]):200j
        ]

        # PCA to the rank actually present (e.g., 1D if manifold is a curve)
        pca = PCA(n_components=2, svd_solver="full")
        E2 = pca.fit_transform(E)

        sig, bkg = E2[ms].T, E2[mb].T     # shapes: (d, n)
        kde_sig = gaussian_kde(sig)       # now covariance is full-rank in reduced space
        kde_bkg = gaussian_kde(bkg)

        z_sig = kde_sig(np.vstack([grid_x.ravel(), grid_y.ravel()]))
        z_bkg = kde_bkg(np.vstack([grid_x.ravel(), grid_y.ravel()]))

        enrichment = np.log((z_sig + 1e-6) / (z_bkg + 1e-6))
        plt.imshow(enrichment.reshape(200,200), extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                origin='lower', cmap='cool')
        plt.title('Signal enrichment map in UMAP space')
        plt.colorbar(label='log(signal/background)')
        plt.show()
    

    
    def pairwise_wasserstein_distance(self, loader, ntheta=20):
        phi_s_list = []
        phi_b_list = []
        features_theta = []
        labels = []
        self.extract_features(loader)
        #with torch.no_grad():
        for i, batch in enumerate(loader):
            _, query, yb = batch
            yb = yb[0].view(-1) if yb[0].ndim > 1 else yb[0]
            s_mask = (yb == 1)
            b_mask = (yb == 0)
            phib = query.phi[0]
            if s_mask.any(): phi_s_list.append(phib[s_mask].cpu())
            else:            phi_s_list.append(torch.empty(0, phib.shape[-1]))
            if b_mask.any(): phi_b_list.append(phib[b_mask].cpu())
            else:            phi_b_list.append(torch.empty(0, phib.shape[-1]))

        phi_list = phi_s_list[:ntheta]
        y_list = labels[:ntheta]
        # assuming D[i, j] contains the Wasserstein distance between phi_i and phi_j
        B = len(phi_list)
        D = np.zeros((B, B))

        for (i, phi_i), (j, phi_j) in itertools.combinations(enumerate(phi_list), 2):
            d = wasserstein_distance(
                phi_i.flatten().cpu().numpy(),
                phi_j.flatten().cpu().numpy()
            )
            D[i, j] = D[j, i] = d

        plt.figure(figsize=(6, 5))
        sns.heatmap(D, annot=False, cmap="viridis")
        plt.title("Pairwise Wasserstein Distances between $\phi$-distributions (signal only)")
        plt.xlabel(r"$\theta_j$")
        plt.ylabel(r"$\theta_i$")
        plt.show()

        phi_list = phi_b_list[:ntheta]
        y_list = labels[:ntheta]
        # assuming D[i, j] contains the Wasserstein distance between phi_i and phi_j
        B = len(phi_list)
        D = np.zeros((B, B))

        for (i, phi_i), (j, phi_j) in itertools.combinations(enumerate(phi_list), 2):
            d = wasserstein_distance(
                phi_i.flatten().cpu().numpy(),
                phi_j.flatten().cpu().numpy()
            )
            D[i, j] = D[j, i] = d

        plt.figure(figsize=(6, 5))
        sns.heatmap(D, annot=False, vmin = 0.0, vmax=0.16,cmap="viridis")
        plt.title("Pairwise Wasserstein Distances between $\phi$-distributions (background only)")
        plt.xlabel(r"$\theta_j$")
        plt.ylabel(r"$\theta_i$")
        plt.show()
    
    def signal_vs_background_wasserstein(self, loader):
        phi_s_list = []
        phi_b_list = []

        self.extract_features(loader)

        #with torch.no_grad():
        for i, batch in enumerate(loader):
            _, query, yb = batch
            yb = yb[0].view(-1) if yb[0].ndim > 1 else yb[0]
            s_mask = (yb == 1)
            b_mask = (yb == 0)
            phib = query.phi[0]
            if s_mask.any(): phi_s_list.append(phib[s_mask].cpu())
            else:            phi_s_list.append(torch.empty(0, phib.shape[-1]))
            if b_mask.any(): phi_b_list.append(phib[b_mask].cpu())
            else:            phi_b_list.append(torch.empty(0, phib.shape[-1]))

        def wdist_1d(a, b):
            a = a.detach().cpu().numpy().ravel()
            b = b.detach().cpu().numpy().ravel()
            return wasserstein_distance(a, b)

        d_sep = []
        for b in range(len(phi_s_list)):
            if phi_s_list[b].numel() == 0 or phi_b_list[b].numel() == 0:
                d_sep.append(np.nan)  # or 0.0
            else:
                d_sep.append(wdist_1d(phi_s_list[b], phi_b_list[b]))

        plt.figure(figsize=(6, 5))
        plt.hist(d_sep)
        plt.title("Within-batch Signal vs Background separability ")
        plt.xlabel(r"Wasserstein Distances")
        plt.ylabel(r"count")
        plt.show()

    def log_signal_over_background_filtered(self):
        E =  np.hstack([self.embeddings["theta"], self.embeddings["phi"]])

        E = np.asarray(E)   # your UMAP coords
        Y = np.asarray(self.labels).ravel()

        ms = (Y == 1)
        mb = (Y == 0)

        nx = ny = 60  # coarser grid (was 160)
        eps = 1e-4

        Hsig, _, _ = np.histogram2d(E[ms,0], E[ms,1], bins=[nx, ny])
        Hbkg, _, _ = np.histogram2d(E[mb,0], E[mb,1], bins=[nx, ny])

        # smoothing to avoid checkerboard
        
        Hsig_s = gaussian_filter(Hsig, sigma=1)
        Hbkg_s = gaussian_filter(Hbkg, sigma=1)

        log_ratio = np.log((Hsig_s + eps) / (Hbkg_s + eps))

        plt.figure(figsize=(7,6))
        plt.imshow(
            log_ratio.T, origin="lower",
            extent=(E[:,0].min(), E[:,0].max(), E[:,1].min(), E[:,1].max()),
            aspect="auto", cmap="coolwarm"
        )
        plt.colorbar(label="log(signal / background)")
        plt.title("Signal enrichment map in UMAP space (smoothed)")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.show()