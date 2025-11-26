import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score
import pandas as pd
import pickle
from resolve.network_architectures.leaf_cache import LeafCache

class XGBoostWrapper(nn.Module):
    """
    PyTorch-friendly wrapper around xgboost with an AE-like API.

    - forward(query_theta, query_phi, **kwargs) -> {"logits": [preds]}
    - fit(...) can take:
        * loader=DataLoader/IterableDataset yielding (context, query, target)
          where query has .theta, .phi and target is a tensor or has .y
        * X, y tensors directly
        * query_theta/query_phi + y/target tensors
    """

    def __init__(
        self,
        config: dict,
        task: str = "regression",        # "regression" or "binary"
        use_parameter_search: bool = False,
        use_leaf_embeddings: int | bool = False,
        **extra_params,
    ):
        super().__init__()

        if task not in ("regression", "binary"):
            raise ValueError(f"task must be 'regression' or 'binary', got {task}")

        self.task = task

        config.update(extra_params)

        if self.task == "regression":
            self.model = xgb.XGBRegressor(**config)
        else:  # binary classification
            self.model = xgb.XGBClassifier(**config)

        self.use_parameter_search = use_parameter_search
        self._fitted = False
        self.booster = None
        # device handling: XGBoost stays on CPU; we only control output device for tensors

        self.cpu_only = True  # hint for trainers
        # XGBoost leaf embedding support

        self.leaf_embed_dim = use_leaf_embeddings                   # embedding size per tree leaf
        self.leaf_embeddings = None               # created after fitting

    # utilities: input & conversion

    @staticmethod
    def _concat_inputs(query_theta: torch.Tensor | None,
                       query_phi: torch.Tensor | None) -> torch.Tensor:
        """Concatenate along last dim; supports 2D (N,D) or 3D (B,T,D)."""
        if query_theta is None and query_phi is None:
            raise ValueError("Provide at least one of query_theta or query_phi.")

        if query_theta is None:
            return query_phi
        if query_phi is None:
            return query_theta
        return torch.cat([query_theta, query_phi], dim=-1)

    # training
    def fit(
        self,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,
    ):
        """
        Fit the XGBoost model.

        Args:
            loader: DataLoader/IterableDataset yielding:
                    - (context, query, target), where
                      query.theta, query.phi are tensors and target is tensor/has .y
            X: pre-concatenated tensor of shape (N,D) or (B,T,D)
            y: target tensor, broadcastable to X's first dims
            query_theta/query_phi: tensors to be concatenated along last dim
            target: alternative handle for y if passed separately
        """

        # Case 1: build X, y from loader batches
        if loader is not None:
            X_parts = []
            y_parts = []

            for i, batch in enumerate(loader):
                # Expect something like: (context, query, target)
                if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                    raise ValueError(
                        "Expected loader batch to be (context, query, target). "
                        f"Got type {type(batch)} with len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}."
                    )
                context, query, target_b = batch

                X_tgt = self._concat_inputs(query.theta, query.phi)
                X_ctx = self._concat_inputs(context.theta, context.phi)
                X_batch = torch.cat([X_tgt,X_ctx],dim=-2)

                X_parts.append(X_batch)
                y_tensor = torch.cat([target_b, context.y], -2)

                # Flatten y to match X flattening later
                y_parts.append(y_tensor.reshape(-1, 1))

            if len(X_parts) == 0:
                raise ValueError("Loader yielded no batches.")

            # concatenate along batch dimension (dim=0)
            X = torch.cat([p.reshape(-1, p.size(-1)) for p in X_parts], dim=0)
            y = torch.cat(y_parts, dim=0).reshape(-1)

        # Case 2: tensors provided directly
        else:
            if X is None:
                X = self._concat_inputs(query_theta, query_phi)

            if y is None and target is not None:
                y = target

            if y is None:
                raise ValueError("Supervised XGBoost requires targets 'y' (or 'target').")

        # Convert X and y to numpy
        X_np = X.squeeze(0).detach().cpu().numpy()
        y_np = y.squeeze(0).detach().cpu().numpy()
        
        X_np, X_val, y_np, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}"
            )

        if self.use_parameter_search == False:
            self.model.fit(
                X_np, y_np,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            xgb_base = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",       # GPU-accelerated histogram algorithm
                device="cuda",
                n_estimators=2000,            # rely on early stopping if you add eval_set
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
            )
            # Define random search parameter space
            param_distributions = {
                "max_depth": [3, 4, 5, 6, 8, 10],
                "min_child_weight": [1, 2, 3, 5, 7, 10],
                "gamma": [0, 0.5, 1.0, 2.0, 5.0],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
            }

            precision_scorer = make_scorer(precision_score, average="binary")

            # RandomizedSearchCV setup
            # n_jobs=1 is safest with GPU
            gsearch = RandomizedSearchCV(
                estimator=xgb_base,
                param_distributions=param_distributions,
                n_iter=30,                 # number of random hyperparam configs to try
                scoring=precision_scorer,  # or 'roc_auc', 'neg_log_loss', etc.
                cv=2,                      # 2-fold CV per config
                verbose=3,
                n_jobs=1,                  # GPU + multiple processes can fight; keep 1
                random_state=42,
            )

            # Fit
            gsearch.fit(X_np, y_np, 
                        eval_set=[(X_val, y_val)],
                        verbose=False)

            print("Best params:", gsearch.best_params_)
            print("Best CV score:", gsearch.best_score_)

            cv_results = gsearch.cv_results_
            scores_df = pd.DataFrame(cv_results).sort_values(by="rank_test_score")

            scores_df.to_csv("./xgb_random_search_results.csv", index=False)

            self.model = gsearch.best_estimator_

        self._fitted = True
        self.booster = self.model.get_booster()

        # Build embedding tables for leaves of each tree
        if self.leaf_embed_dim:

            df = self.booster.trees_to_dataframe()  # no data needed, pure model metadata

            # Leaves are rows where Feature == 'Leaf'
            leaf_nodes = df[df["Feature"] == "Leaf"]

            # For each tree, get the maximum node id among leaf nodes.
            max_node_id_per_tree = leaf_nodes.groupby("Tree")["Node"].max().sort_index()

            # Embedding size per tree = max_node_id + 1
            num_nodes_per_tree = (max_node_id_per_tree.values + 1).astype(int)

            self.leaf_embeddings = nn.ModuleList(
                [nn.Embedding(int(n_nodes), self.leaf_embed_dim) for n_nodes in num_nodes_per_tree]
            )

        return self

    @staticmethod
    def _to_2d_numpy(x: torch.Tensor) -> tuple[np.ndarray, tuple]:
        """
        Flatten (B,T,D)->(B*T,D) or (N,D)->(N,D), detach to CPU numpy.
        Returns (np_array, original_shape_tuple)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x2 = x.reshape(B * T, D)
            return x2.detach().cpu().numpy(), (B, T, D)
        elif x.dim() == 2:
            N, D = x.shape
            return x.detach().cpu().numpy(), (N, D)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

    @staticmethod
    def _from_2d_numpy(preds: np.ndarray, original_shape: tuple) -> torch.Tensor:
        """Restore predictions to (B,T,1) or (N,1)."""
        if len(original_shape) == 3:  # (B,T,D)
            B, T, _ = original_shape
            return torch.from_numpy(preds.astype(np.float32)).reshape(B, T, 1)
        else:  # (N,D)
            N, _ = original_shape
            return torch.from_numpy(preds.astype(np.float32)).reshape(N, 1)

    @staticmethod
    def _targets_to_1d_numpy(y: torch.Tensor) -> np.ndarray:
        """
        Convert y with shape (B,T,1), (B,T), (N,1), or (N,) to (N_total,) numpy.
        """
        y = y.detach().cpu()
        if y.dim() == 3:
            B, T, C = y.shape
            if C != 1:
                raise ValueError(f"Expected last dim=1 for 3D targets, got {C}")
            return y.reshape(B * T).numpy()
        elif y.dim() == 2:
            N, C = y.shape
            if C == 1:
                return y.reshape(N).numpy()
            else:
                raise ValueError(f"Multi-output targets not supported, got shape {tuple(y.shape)}")
        elif y.dim() == 1:
            return y.numpy()
        else:
            raise ValueError(f"Unsupported target shape {tuple(y.shape)}")
    # inference
    def predict(self, X_torch):
        X_np, original_shape = self._to_2d_numpy(X_torch)
        #self.booster.set_param({"predictor": "cpu_predictor"})
        preds = self.booster.inplace_predict(X_np).astype(np.float32)
        preds_t = self._from_2d_numpy(preds, original_shape)  
        return preds_t.to(X_torch.device)

    
    # encoding
    @torch.no_grad()
    def encode(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Core function used by memory bank (or anyone else).

        X: (N,D) or (B,T,D) tensor on any device
        Returns:
            scores:   (N, out_dim)
            leaf_emb: (N, leaf_embed_dim)
        """
        if not self._fitted:
            raise RuntimeError("XGBoostWrapper not fitted. Call fit() first.")

        X_np, original_shape = self._to_2d_numpy(X)

        # XGB scores
        scores_np = self.booster.inplace_predict(X_np).astype("float32")
        scores_t = torch.from_numpy(scores_np)

        if scores_t.ndim == 1:
            scores_t = scores_t.unsqueeze(-1)        # (N,1)

        scores_t = scores_t.to(X.device)

        # leaf embeddings
        if self.leaf_embeddings is None:
            leaf_emb = torch.empty(scores_t.shape[0], 0, device=X.device)
        else:
            emb_device = next(self.leaf_embeddings[0].parameters()).device
            leaf_arr = self.model.apply(X_np).astype("int64")          # (N, n_trees)
            leaf_ids = torch.from_numpy(leaf_arr).to(emb_device)       # (N, n_trees)

            embeds = []
            for t, emb_layer in enumerate(self.leaf_embeddings):
                ids_t = leaf_ids[:, t]          # (N,)
                embeds.append(emb_layer(ids_t)) # (N, leaf_embed_dim)

            leaf_emb = torch.stack(embeds, dim=1).sum(dim=1)           # (N, leaf_embed_dim)
            leaf_emb = leaf_emb.to(X.device)

        return scores_t, leaf_emb

    def forward(self, query_theta, query_phi, **kwargs):
        """
        Simple, stateless forward. No memory, no indices.
        """
        X = self._concat_inputs(query_theta, query_phi)
        scores, leaf_emb = self.encode(X)
        B, T, _ = query_theta.shape
        scores = scores.view(B, T, -1)
        leaf_emb = leaf_emb.view(B, T, -1) if leaf_emb.numel() > 0 else leaf_emb

        return {
            "logits": [scores],
            "leaf_embeddings": leaf_emb,
        }
    
    def save(self, path):
        # Save sklearn model
        
        with open(path + "xgb.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # Save booster
        booster = self.model.get_booster()
        booster.save_model(path + "booster.json")

        torch.save(self.state_dict(), path+"embeddings.pt")
        print(f"Saved XGBClassifier to {path}xgb.pkl and booster to {path}booster.json")

    def load(self, path):
        with open(path + "xgb.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.booster = self.model.get_booster()
        state = torch.load(path+"embeddings.pt", map_location="cpu")
        self.load_state_dict(state)
        print(f"Loaded XGBClassifier from {path}xgb.pkl")

class XGBWithLeafCache(XGBoostWrapper):
    """
    XGBoostWrapper + optional LeafCache.

    Behaves exactly like XGBoostWrapper if:
      - no LeafCache is attached, or
      - forward() is called without query_idx.

    Uses LeafCache when:
      - leaf_cache is not None, AND
      - query_idx is provided.
    """

    def __init__(
        self,
        config: dict,
        task: str = "regression",
        out_dim: int = 1,
        use_parameter_search: bool = False,
        use_leaf_embeddings: int | bool = False,
        num_samples: int | None = None,
        device: torch.device = torch.device("cpu"),
        **extra_params,
    ):
        super().__init__(
            config=config,
            task=task,
            out_dim=out_dim,
            use_parameter_search=use_parameter_search,
            use_leaf_embeddings=use_leaf_embeddings,
            **extra_params,
        )
        self.device = device
        self.out_dim = out_dim
        self.leaf_cache = None

        if num_samples is not None and self.leaf_embed_dim:
            self.leaf_cache = LeafCache(
                num_samples=num_samples,
                out_dim=self.out_dim,
                leaf_embed_dim=self.leaf_embed_dim,
                device=self.device,
            )

    def enable_leaf_cache(self, num_samples: int, device: torch.device | None = None):
        """
        Lazily attach / reattach a LeafCache after init.
        """

        if device is None:
            device = next(self.parameters()).device
        if not self.leaf_embed_dim:
            raise ValueError("leaf_embed_dim is 0/False, cannot create LeafCache.")

        self.leaf_cache = LeafCache(
            num_samples=num_samples,
            out_dim=self.out_dim,
            leaf_embed_dim=self.leaf_embed_dim,
            device=device,
        )

    def forward(self, query_theta, query_phi, query_idx=None, **kwargs):
        """
        If query_idx and leaf_cache present → use memory.
        Else → fall back to parent (stateless) forward.
        """
        # no cache or no indices: behave like plain XGBoostWrapper
        if self.leaf_cache is None or query_idx is None:
            return super().forward(query_theta, query_phi, **kwargs)

        # with cache
        X = self._concat_inputs(query_theta, query_phi)  # (B,T,D)
        B, T, _ = X.shape

        idx_flat = query_idx.view(-1)
        scores_flat, leaf_flat = self.leaf_cache.query(self.encode, X, idx_flat)

        preds = scores_flat.view(B, T, -1)
        leaf_emb = leaf_flat.view(B, T, -1)

        return {
            "logits": [preds],
            "leaf_embeddings": leaf_emb,
        }
    
    def save(self, path: str):
        with open(path + "xgb.pkl", "wb") as f:
            pickle.dump(self.model, f)

        booster = self.model.get_booster()
        booster.save_model(path + "booster.json")

        state = self.state_dict()
        # drop all leaf_cache.* entries from the state dict
        state = {k: v for k, v in state.items() if not k.startswith("leaf_cache.")}
        torch.save(state, path + "embeddings.pt")

        self.leaf_cache.save_cache(path + "leaf_cache.pt")

    def load(self, path: str):
        with open(path + "xgb.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.booster = self.model.get_booster()

        state = torch.load(path + "embeddings.pt", map_location="cpu")
        self.load_state_dict(state)

        if self.leaf_cache is not None:
            self.leaf_cache.load_cache(path + "leaf_cache.pt")