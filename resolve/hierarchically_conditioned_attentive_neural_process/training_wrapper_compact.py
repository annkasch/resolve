
import os, time
import gc
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import h5py
# Optional local deps (kept to preserve your original interface)
try:
    from .zarr_helper import ZarrPredWriter
except Exception:
    ZarrPredWriter = None  # type: ignore

try:
    from .data_generator import BatchFormatter
except Exception:
    BatchFormatter = None  # type: ignore

# Keep your default criterion name (replace with your own impl if available)
try:
    from .losses import AsymmetricFocalWithFPPenalty  # your custom loss
except Exception:
    AsymmetricFocalWithFPPenalty = None  # type: ignore

import subprocess

def get_git_hash(short=True):
    try:
        args = ["git", "rev-parse", "HEAD"]
        if short:
            args.insert(2, "--short")
        return subprocess.check_output(args).decode("utf-8").strip()
    except Exception:
        return "unknown"



def _device() -> torch.device:
    # Prefer MPS if available, else CUDA, else CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_detach_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _is_binary_range(target_range: Tuple[float, float]) -> bool:
    lo, hi = target_range
    return lo >= 0.0 and hi <= 1.0


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_binary: bool,
) -> Dict[str, float]:
    """Compact metrics with safe fallbacks for edge cases."""
    metrics: Dict[str, float] = {}
    if is_binary:
        # Probabilities -> labels with 0.5 threshold (customize if needed)
        y_prob = y_pred.reshape(-1)
        y_hat = (y_prob >= 0.5).astype(np.int32)
        y_true_i = y_true.reshape(-1).astype(np.int32)

        # Robustness against single-class edge cases
        try:
            metrics["accuracy"] = float(accuracy_score(y_true_i, y_hat))
        except Exception:
            metrics["accuracy"] = float("nan")
        try:
            metrics["precision"] = float(precision_score(y_true_i, y_hat, zero_division=0))
        except Exception:
            metrics["precision"] = float("nan")
        try:
            metrics["recall"] = float(recall_score(y_true_i, y_hat, zero_division=0))
        except Exception:
            metrics["recall"] = float("nan")
        try:
            metrics["f1"] = float(f1_score(y_true_i, y_hat, zero_division=0))
        except Exception:
            metrics["f1"] = float("nan")
        # AUCs require both classes present
        if len(np.unique(y_true_i)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_i, y_prob))
            except Exception:
                metrics["roc_auc"] = float("nan")
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true_i, y_prob))
            except Exception:
                metrics["pr_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    else:
        y_true_f = y_true.reshape(-1)
        y_pred_f = y_pred.reshape(-1)
        metrics["mae"] = float(mean_absolute_error(y_true_f, y_pred_f))
        metrics["mse"] = float(mean_squared_error(y_true_f, y_pred_f))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        try:
            metrics["r2"] = float(r2_score(y_true_f, y_pred_f))
        except Exception:
            metrics["r2"] = float("nan")
    return metrics


class Trainer:
    """Compact, efficient trainer with early stopping and checkpointing.
    Assumptions about your dataset:
      - dataset.train_loader() and dataset.val_loader() return PyTorch DataLoaders
      - batches are dicts understood by BatchFormatter (if provided)
      - dataset.config_file["simulation_settings"]["target_range"] exists
    """

    def __init__(self, model, dataset, normalizer=None, epochs: int = 10):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.epoch_start = 0

        # Binary vs regression detection
        self.target_range = self.dataset.config_file["simulation_settings"]["target_range"]
        self.is_binary = _is_binary_range(self.target_range)

        # Keep your preferred loss if available, else BCE as reasonable defaults
        if AsymmetricFocalWithFPPenalty is not None and self.is_binary:
            self.criterion = AsymmetricFocalWithFPPenalty()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss() if self.is_binary else torch.nn.HuberLoss()
        # Formatter (if available)
        self.formatter = BatchFormatter(self.dataset.parameters, context_ratio=getattr(self.dataset, "_context_ratio", None), normalizer=normalizer) if BatchFormatter else None

        # For logging last epoch metrics
        self.metrics_train: Dict[str, float] = {}
        self.metrics_val: Dict[str, float] = {}

    def _forward_batch(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format and forward a single batch. Expects keys consistent with your data pipeline."""
        context, query, targets = self.formatter(batch)
        logit, kl_term = self.model(context.theta, context.phi, context.y, query.theta, query.phi)
        return logit, kl_term, targets

    def _run_epoch(self, loader, optimizer=None, train: bool = True, desc: str = "train") -> Tuple[float, np.ndarray, np.ndarray]:
        device = next(self.model.parameters()).device
        self.model.train(train)
        running_loss = 0.0
        y_true_all, y_pred_all = [], []

        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            # Move tensors to device
            def _to_dev(t):
                return t.to(device) if isinstance(t, torch.Tensor) else t
            batch = {k: _to_dev(v) for k, v in (batch.items() if isinstance(batch, dict) else enumerate(batch))}

            if train:
                optimizer.zero_grad(set_to_none=True)

            logit, kl_term, targets = self._forward_batch(batch)

            # For BCEWithLogits, criterion expects logits; keep outputs as-is
            loss = self.criterion(logit, targets) + kl_term

            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.detach().cpu())
            y_true_all.append(_safe_detach_numpy(targets))
            # For binary, store probabilities for metrics; for regression, raw outputs
            if self.is_binary:
                y_pred_all.append(torch.sigmoid(logit).detach().cpu().numpy())
            else:
                y_pred_all.append(_safe_detach_numpy(logit))

            pbar.set_postfix(loss=f"{running_loss/len(y_true_all):.4f}")

        y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
        y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
        avg_loss = running_loss / max(1, len(y_true_all))
        return avg_loss, y_true, y_pred

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        writer=None,
        monitor: str = "pr_auc",  # for binary; for regression we'll silently map to 'rmse'
        mode: str = "max",
        patience: int = 10,
        ckpt_dir: str = "./checkpoints",
        ckpt_name: str = "best.pt",
    ) -> Dict[str, float]:
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = os.path.join(ckpt_dir, ckpt_name)

        device = _device()
        self.model.to(device)

        # Initialize best score based on objective
        if not self.is_binary and monitor.lower() in {"pr_auc", "roc_auc"}:
            monitor = "rmse"
            mode = "min"

        best_score = -float("inf") if mode == "max" else float("inf")
        no_improve = 0

        for epoch in range(self.epoch_start, self.epoch_start + self.epochs):
            # TRAIN
            dataloader = self.dataset.set_loader(mode="train")
            train_loss, y_true_tr, y_pred_tr = self._run_epoch(dataloader, optimizer, train=True, desc=f"train {epoch}")
            m_tr = _compute_metrics(y_true_tr, y_pred_tr, self.is_binary)
            m_tr["loss"] = train_loss
            self.metrics_train = m_tr
            self.dataset.close_loader()

            # VAL
            dataloader = self.dataset.set_loader(mode="validate")
            with torch.inference_mode():
                val_loss, y_true_v, y_pred_v = self._run_epoch(dataloader, optimizer=None, train=False, desc=f"val   {epoch}")
            m_v = _compute_metrics(y_true_v, y_pred_v, self.is_binary)
            m_v["loss"] = val_loss
            self.metrics_val = m_v
            self.dataset.close_loader()

            # Log
            if writer:
                for k, v in m_tr.items():
                    writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in m_v.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            # Early stopping / checkpointing
            score = m_v.get(monitor.lower(), m_v.get(monitor, float("nan")))
            improved = (score > best_score) if mode == "max" else (score < best_score)
            if improved:
                best_score = score
                no_improve = 0
                torch.save({"epoch": epoch, "model_state": self.model.state_dict()}, best_ckpt)
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            # Memory hygiene
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"best_score": float(best_score), "monitor": monitor, "mode": mode, "epochs_ran": epoch - self.epoch_start + 1}


    
    @torch.inference_mode()
    def predict(self, dataset_predict):
            """
            Run the model in prediction mode over the given dataset.
            Returns:
                - mu_all: np.ndarray of predicted means
                - sigma_all: np.ndarray of predicted standard deviations
                - target_y_all: np.ndarray of ground truth targets
                - features_all: np.ndarray of unnormalized input features
            """
            device = torch.device("cpu")
            self.model.to(device)
            self.model.eval()
            
            num_workers = dataset_predict.config_file["cnp_settings"]["dataloader_number_of_workers"]
            
            theta_size = dataset_predict.parameters["theta"]["size"]
            phi_size = dataset_predict.parameters["phi"]["size"]
            y_size = dataset_predict.parameters["target"]["size"]
            
            y_pred = [np.zeros((0, y_size)) for _ in range(num_workers)]
            y_true = [np.zeros((0, y_size)) for _ in range(num_workers)]
            theta = [np.zeros((0, theta_size)) for _ in range(num_workers)]
            phi = [np.zeros((0, phi_size)) for _ in range(num_workers)]
            loss = [0.0 for _ in range(num_workers)]

            dataloader = dataset_predict.set_loader(mode=dataset_predict.mode)
            # Progress bar for files
            pbar = tqdm(total=len(dataloader.dataset.files), desc="Processing files", unit="file")

            for batch, worker_id, file_idx, file_completed in tqdm(dataloader, desc="predict", leave=False):

                context, query, targets = self.formatter(batch, context_is_subset=self.dataset.config_file["cnp_settings"]["context_is_subset"])
                logit, kl_term = self.model(context.theta, context.phi, context.y, query.theta, query.phi)

                y_pred[worker_id]= np.concatenate([y_pred[worker_id], torch.sigmoid(logit)[0].cpu().numpy()], axis=0)  
                y_true[worker_id]= np.concatenate([y_true[worker_id], targets[0].cpu().numpy()], axis=0)

                theta[worker_id]= np.concatenate([theta[worker_id], self.formatter.x_query[:,:theta_size].cpu().numpy()], axis=0)
                phi[worker_id]= np.concatenate([phi[worker_id], self.formatter.x_query[:,theta_size:].cpu().numpy()], axis=0)

                loss[worker_id] += self.criterion(logit, targets) + kl_term

                if file_completed:

                    phi_idx   = dataset_predict.parameters["phi"]["selected_indices"]
                    theta_idx   = dataset_predict.parameters["theta"]["selected_indices"]
                    target_idx   = dataset_predict.parameters["target"]["selected_indices"]

                    with h5py.File(dataset_predict.files[file_idx], "a") as f:
                        d_phi   = np.array(f[dataset_predict.parameters["phi"]["key"]][:,phi_idx])
                        d_theta   = np.array(f[dataset_predict.parameters["theta"]["key"]][theta_idx]).reshape(-1,theta_size)
                        d_target   = np.array(f[dataset_predict.parameters["target"]["key"]][:, target_idx])

                        theta_ok = np.allclose(d_theta,  np.asarray(theta[worker_id], dtype=d_theta.dtype)[None, :], rtol=1e-6, atol=1e-8)
                        phi_ok   = np.allclose(d_phi,    np.asarray(phi[worker_id],   dtype=d_phi.dtype)[None, :],   rtol=1e-6, atol=1e-8)
                        y_ok     = np.allclose(d_target, np.asarray(y_true[worker_id],dtype=d_target.dtype)[None, :], rtol=1e-6, atol=1e-8)

                        if not (theta_ok and phi_ok and y_ok):
                            print("Warning: Data mismatch between loaded file and accumulated arrays!")
                            print(dataset_predict.files[file_idx])
                        else:
                            model_name = self.model.__class__.__name__
                            version = dataset_predict.config_file["path_settings"]["version"]

                            if f"{dataset_predict.mode}/{model_name}_{version}" in f:
                                del f[f"{dataset_predict.mode}/{model_name}_{version}"]
                            grp = f.require_group(f"{dataset_predict.mode}/{model_name}_{version}")

                            d_y_pred = grp.create_dataset("y_pred", data=y_true[worker_id], compression="gzip", chunks=True)
                            # d_std = grp.create_dataset("std", data=y_hat_std, compression="gzip", chunks=True)
                            d_norm_mean = grp.create_dataset("norm_mean", data=self.formatter.norm.scaler.mean_, compression="gzip", chunks=True)
                            d_norm_scale = grp.create_dataset("norm_scale", data=self.formatter.norm.scaler.scale_, compression="gzip", chunks=True)
                            d_loss = grp.create_dataset("loss", data=float(loss[worker_id]))
                            metric = _compute_metrics(y_true[worker_id], y_pred[worker_id], self.is_binary)

                            for n, data in metric.items():
                                grp.create_dataset(n, data=data)

                            # provenance in attributes
                            grp.attrs["created_at"]   = time.strftime("%Y-%m-%d %H:%M:%S") 
                            grp.attrs["model_name"]   = model_name
                            grp.attrs["version"]      = version
                            grp.attrs["git_hash"]     = get_git_hash(short=True)
                            grp.attrs["phi_label"] = str(self.dataset.parameters["phi"]["selected_labels"])
                            grp.attrs["theta_label"] = str(self.dataset.parameters["theta"]["selected_labels"])
                            grp.attrs["target_label"] = str(self.dataset.parameters["target"]["selected_labels"])

                            


                    y_pred[worker_id]=np.zeros((0, y_size))
                    y_true[worker_id]=np.zeros((0, y_size))
                    theta[worker_id]=np.zeros((0, theta_size))
                    phi[worker_id]=np.zeros((0, phi_size))
                    
                    pbar.update(1)  # move progress bar by one file
            pbar.close()
            dataset_predict.close_loader()
                        
