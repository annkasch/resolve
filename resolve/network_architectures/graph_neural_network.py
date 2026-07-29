import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, add_self_loops
import hnswlib


class GNNBinaryClassifier(nn.Module):
    """
    Classic GCN-style GNN for binary classification that matches the interface of
    NeuralDensityRatioEstimator:

        __init__(d_theta, d_phi, d_y, ...)
        forward(query_theta, query_phi, **kwargs) -> {"logits": [logits]}

    - One node per (theta, phi) pair.
    - One logit per node (binary classification with BCEWithLogitsLoss).
    - You must pass `edge_index` in **kwargs.
    """

    def __init__(
        self,
        d_theta,
        d_phi,
        d_y,
        d_model=64,
        encoder_hidden=[128, 128],
        gnn_hidden=[64, 64],
        mode="concat",
        theta_embed_dim=None,
        dropout=0.5,
        use_layernorm=True,
    ):
        super().__init__()

        # --- Feature encoder: (theta, phi) -> node embedding of size d_model ---
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi,
            y_dim=None,
            theta_in_dim=d_theta,
            hidden=encoder_hidden,
            out_dim=d_model,
            mode=mode,
            theta_embed_dim=theta_embed_dim,
            use_layernorm=use_layernorm,
        )

        # --- Classic GCN stack ---
        dims = [d_model] + list(gnn_hidden)
        self.convs = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.dropout = nn.Dropout(dropout)

        # --- Classification head: node embedding -> logit ---
        # d_y should be 1 for binary classification, but we keep it general
        self.head = nn.Linear(dims[-1], d_y)

    def forward(
        self,
        query_theta,
        query_phi,
        **kwargs,
    ):
        """
        Args:
            query_theta: (N, d_theta)
            query_phi:   (N, d_phi)

        Kwargs (required):
            edge_index: (2, E) PyG edge index

        Returns:
            {"logits": [logits]} with logits of shape (N, d_y)
        """

        if self.edge_index is None:
            raise ValueError(
                "ClassicGNNBinaryClassifier.forward expects 'edge_index' in **kwargs."
            )

        # 1) Encode (theta, phi) into node features
        h = self.qry_enc(theta=query_theta, phi=query_phi)  # (N, d_model)

        # 2) Classic GCN layers: Conv -> ReLU -> Dropout
        for conv in self.convs:
            h = conv(h, self.edge_index)   # message passing
            h = F.relu(h)
            h = self.dropout(h)

        # 3) Node-wise logits
        logits = self.head(h)  # (N, d_y)

        return {"logits": [logits]}
    
    def fit(
        self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,
        k=5, metric="euclidean", add_loops=True
    ):
        if loader is not None:
            theta, phi, y, _file_indices = (
                loader.dataset.materialized_tensors(
                    "GNNBinaryClassifier.fit"
                )
            )
            # Move to CPU and build feature matrix X = [theta | phi]
            if theta.device != torch.device("cpu"):
                theta_np = theta.cpu().numpy()
            else:
                theta_np = theta.numpy()

            if phi.device != torch.device("cpu"):
                phi_np = phi.cpu().numpy()
            else:
                phi_np = phi.numpy()

            X = np.concatenate([theta_np, phi_np], axis=-1)  # shape (N, d_theta + d_phi)
        self.edge_index = self.build_rare_event_graph_hnsw(X, y)

    '''
    @torch.no_grad()
    def build_global_edge_index(self,
        X: torch.Tensor,
        k: int = 5,
        metric: str = "euclidean",
        add_loops: bool = True,
    ) -> torch.Tensor:
        """
        Build a *global* kNN graph for a GNN, using all nodes in (theta, phi).
        This is meant to be called ONCE at the beginning of training.
        Args
        ----
        theta : (N, d_theta) torch.Tensor
            Global theta features (e.g., design / condition variables).
        phi   : (N, d_phi) torch.Tensor
            Global phi features (e.g., event / instance features).
        k     : int
            Number of neighbors (excluding self). Effective degree ~ 2*k after
            symmetrization.
        metric : str
            Metric for sklearn NearestNeighbors ("euclidean", "cosine", etc.).
        add_loops : bool
            If True, add self-loops (recommended for GCN/GraphSAGE stability).

        Returns
        -------
        edge_index : LongTensor of shape (2, E)
            Undirected edge_index over all N nodes, suitable for PyG.
        """
        
        N = X.shape[0]

        # kNN search in this feature space
        #    neighbors include the point itself as the first neighbor; we drop it.
        k_eff = min(k + 1, N)
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric=metric)
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)  # indices: (N, k_eff)

        # Drop self (first neighbor)
        indices = indices[:, 1:]   # (N, k)

        # 3) Build directed edges i -> j for each neighbor
        src = np.repeat(np.arange(N), indices.shape[1])
        dst = indices.reshape(-1)
        edge_index = np.stack([src, dst], axis=0)  # (2, E_directed)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # 4) Make undirected, remove duplicates
        edge_index = to_undirected(edge_index, num_nodes=N)

        # 5) Optional self-loops
        if add_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        return edge_index
    '''

    @torch.no_grad()
    def build_rare_event_graph_hnsw(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: int = 5,
        max_nodes: int = 200_000,
        neg_pos_ratio: int = 50,
        space: str = "l2",          # or "cosine"
        ef_construction: int = 200,
        M: int = 32,
        ef: int = 200,
        add_loops: bool = True,
        seed: int = 42,
    ):
        """
        Build a scalable kNN graph for a rare-event dataset using HNSW.

        Strategy:
        - Keep ALL positive samples.
        - Sample up to `neg_pos_ratio * n_pos` negatives.
        - Cap total nodes at `max_nodes`.
        - Build an approximate kNN graph (HNSW) over this subset only.

        Args
        ----
        theta : (N, d_theta) tensor
        phi   : (N, d_phi)   tensor
        y     : (N,) or (N,1) tensor of labels in {0,1}
        k     : int, number of neighbors (excluding self)
        max_nodes : int, hard cap on subset size
        neg_pos_ratio : int, max #negatives = neg_pos_ratio * #positives
        space : str, "l2" or "cosine" for hnswlib
        ef_construction, M, ef : HNSW hyperparameters
        add_loops : whether to add self-loops
        seed : RNG seed for negative subsampling

        Returns
        -------
        edge_index : LongTensor (2, E)
            Graph edges over the SUBSET of nodes.
        subset_idx : LongTensor (N_sub,)
            Indices into the original dataset for the nodes in this graph.
        """

        # --- 0) Prep labels ---
        if y.ndim == 2:
            y_flat = y.squeeze(-1)
        else:
            y_flat = y
        y_flat = y_flat.long()

        N = X.shape[-2]

        # --- 1) Build subset indices: all positives + sampled negatives ---
        pos_idx = torch.nonzero(y_flat == 1, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(y_flat == 0, as_tuple=False).view(-1)

        n_pos = pos_idx.numel()
        n_neg = neg_idx.numel()

        if n_pos == 0:
            raise ValueError("No positive samples found in y; rare-event graph construction won't work.")

        # target negatives = min(neg_pos_ratio * n_pos, n_neg, max_nodes - n_pos)
        max_negs_from_ratio = neg_pos_ratio * n_pos
        max_negs_from_cap = max_nodes - n_pos
        target_neg = min(max_negs_from_ratio, n_neg, max_negs_from_cap)

        if target_neg <= 0:
            # extremely imbalanced or tiny max_nodes
            target_neg = min(n_neg, max_nodes - n_pos)

        # sample negatives
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        neg_idx_cpu = neg_idx.cpu()
        perm = torch.randperm(n_neg, generator=rng)
        neg_sampled = neg_idx_cpu[perm[:target_neg]]

        # combine positives + sampled negatives
        subset_idx = torch.cat([pos_idx.cpu(), neg_sampled], dim=0)
        subset_idx = subset_idx.unique()  # just in case

        n_sub = subset_idx.numel()
        print(f"[build_rare_event_graph_hnsw] N={N}, positives={n_pos}, using subset size={n_sub}")

        # --- 2) Extract subset features and move to numpy ---
        X_sub = X[subset_idx].astype(np.float32)  # (n_sub, d)

        # --- 3) Build HNSW index on subset ---
        n_sub, dim = X_sub.shape
        p = hnswlib.Index(space=space, dim=dim)
        p.init_index(max_elements=n_sub, ef_construction=ef_construction, M=M)
        p.set_num_threads(0)
        labels = np.arange(n_sub)
        p.add_items(X_sub, labels)
        p.set_ef(ef)

        # --- 4) Query k+1 neighbors (includes self) ---
        I, _ = p.knn_query(X_sub, k=min(k + 1, n_sub))
        # drop self (assume first is self)
        I = I[:, 1:]  # (n_sub, k_eff)

        src = np.repeat(np.arange(n_sub), I.shape[1])
        dst = I.reshape(-1)

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # make undirected
        edge_index = to_undirected(edge_index, num_nodes=n_sub)

        # add self-loops if desired
        if add_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=n_sub)

        # map subset_idx back to original device (if you care)
        subset_idx = subset_idx

        return edge_index, subset_idx

    def save(self, path):
        torch.save(self.state_dict(), path + "_model.pth")
