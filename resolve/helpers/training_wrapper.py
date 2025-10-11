
import os, time
import gc
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import math
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
import dataclasses
from ..utilities import utilities as utils
from collections.abc import Mapping, Sequence
import time, torch

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

def _to_dev(obj, device, *, non_blocking=False):
            """Recursively move all tensors inside obj to device.
            Supports: Tensor, dict, list/tuple, dataclass, NamedTuple, and objects with .to().
            """
            # Tensors
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=non_blocking)

            # Mappings (dict-like)
            if isinstance(obj, Mapping):
                return obj.__class__({k: _to_dev(v, device, non_blocking=non_blocking) for k, v in obj.items()})

            # Sequences (but not str/bytes); rebuild tuples to keep tuple type
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # NamedTuple
                    return obj.__class__(*[_to_dev(v, device, non_blocking=non_blocking) for v in obj])
                return obj.__class__([_to_dev(v, device, non_blocking=non_blocking) for v in obj])

            # Dataclasses (frozen or not)
            if dataclasses.is_dataclass(obj):
                values = {f.name: _to_dev(getattr(obj, f.name), device, non_blocking=non_blocking) for f in dataclasses.fields(obj)}
                return dataclasses.replace(obj, **values)

            # Objects exposing .to()
            to = getattr(obj, "to", None)
            if callable(to):
                try:
                    return to(device)  # e.g., user-defined containers with a .to()
                except TypeError:
                    pass  # .to signature not compatible; fall through

            # Leave other types as-is
            return obj

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

    def __init__(self, model, dataset, epochs: int = 10):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.epoch_start = 0
        self._report = 1
        self.bce = nn.BCELoss()

        # Binary vs regression detection
        self.target_range = self.dataset.config_file["simulation_settings"]["target_range"]
        self.is_binary = _is_binary_range(self.target_range)

        # Keep your preferred loss if available, else BCE as reasonable defaults
        if AsymmetricFocalWithFPPenalty is not None and self.is_binary:
            self.criterion = AsymmetricFocalWithFPPenalty()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss() if self.is_binary else torch.nn.HuberLoss()
        # Formatter (if available)

        # For logging last epoch metrics
        self.metrics: Dict[str, float] = {}

    def _forward_batch(self, batch: dict, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format and forward a single batch. Expects keys consistent with your data pipeline."""
        # unpack the batch:
        context, query, targets = batch

        # move everything to device
        targets = _to_dev(targets, device)
        context = _to_dev(context, device)
        query   = _to_dev(query, device)
        output = self.model(
            context.theta, context.phi, context.y,
            query.theta, query.phi, target_y=targets
        )

        return output, targets

    def _run_epoch(self, loader, optimizer=None, train: bool = True, desc: str = "train") -> Tuple[float, np.ndarray, np.ndarray]:
        device = next(self.model.parameters()).device
        self.model.train(train)
        running_loss = 0.0
        y_true_all, y_pred_all = [], []
        
        pbar = tqdm(loader, total=math.ceil(len(loader) / loader.dataset.batch_size), desc=desc, leave=True)
        for batch in pbar:

            if train:
                optimizer.zero_grad(set_to_none=True)
            output, targets = self._forward_batch(batch, device)

            logit = output.get("logits", None)
            kl_term = output.get("kl_term", 0.0)
            add_loss = output.get("loss", 0.0)

            loss = self.criterion(logit, targets) + kl_term + add_loss

            if train:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.detach().cpu())
            y_true_all.append(_safe_detach_numpy(targets).reshape(-1))
            # For binary, store probabilities for metrics; for regression, raw outputs
            if self.is_binary and self.model._get_name()!= 'ConditionalNeuralProcess':
                y_pred_all.append(torch.sigmoid(logit[0]).detach().cpu().numpy().reshape(-1))
            else:
                y_pred_all.append(_safe_detach_numpy(logit[0]).reshape(-1))
            
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
            dataloader = self.dataset.set_loader("train")
            train_loss, y_true_tr, y_pred_tr = self._run_epoch(dataloader, optimizer, train=True, desc=f"train {epoch+1}/{self.epoch_start + self.epochs}")
            m_tr = _compute_metrics(y_true_tr, y_pred_tr, self.is_binary)
            m_tr["loss"] = train_loss
            self.metrics["train"] = m_tr

            # Log
            if writer and epoch % self._report == 0:
                for k, v in m_tr.items():
                    writer.add_scalar(f"train/{k}", v, epoch+1)
                fig = utils.plot(y_pred_tr.reshape(-1, 1), y_true_tr.reshape(-1, 1), it=epoch+1)
                writer.add_figure(f'plot/train', fig, global_step=epoch+1)

            # Early stopping / checkpointing
            score = self.evaluate(writer=writer, dataset_name="validate", monitor=monitor, epoch=epoch+1)
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

        self.metrics["best_model"]={"best_score": float(best_score), "monitor": monitor, "mode": mode, "epochs_ran": epoch - self.epoch_start + 1}
        return self.metrics["best_model"]

    @torch.inference_mode()
    def evaluate(
        self,
        writer=None,
        dataset_name="validate",
        monitor: str = "pr_auc",  # for binary; for regression we'll silently map to 'rmse'
        epoch: int = 0,
    ) -> Dict[str, float]:

        device = _device()
        self.model.to(device)

        dataloader = self.dataset.set_loader(dataset_name)
        with torch.inference_mode():
            loss, y_true_v, y_pred_v = self._run_epoch(dataloader, optimizer=None, train=False, desc=f"{dataset_name} {epoch+1}/{self.epoch_start + self.epochs}")
        m_v = _compute_metrics(y_true_v, y_pred_v, self.is_binary)
        m_v["loss"] = loss
        self.metrics[dataset_name] = m_v

        # Log
        if writer and epoch % self._report == 0.:
            for k, v in m_v.items():
                writer.add_scalar(f"{dataset_name}/{k}", v, epoch)
            fig = utils.plot(y_pred_v.reshape(-1, 1), y_true_v.reshape(-1, 1), it=epoch)
            writer.add_figure(f'plot/{dataset_name}', fig, global_step=epoch)

        score = m_v.get(monitor.lower(), m_v.get(monitor, float("nan")))
        return score

    @torch.inference_mode()
    def predict(self, dataset_name="predict", monitor="pr_auc",writer=None):
            """
            Run the model in prediction mode over the given dataset.
            Processes data file by file and saves predictions back to the same files.
            """
            device = _device()
            self.model.to(device)
            self.model.eval()
            
            # Get dimensions from dataset parameters
            sizes = {k: self.dataset.parameters[k]["size"] for k in ["theta", "phi", "target"]}
            worker_id = 0  # Since we process one file at a time, we can use a single worker
            
            # Initialize data collectors
            collectors = {
                "y_pred": np.zeros((0, sizes["target"])),
                "y_err": np.zeros((0, sizes["target"])),
                "y_true": np.zeros((0, sizes["target"])),
                "theta": np.zeros((0, sizes["theta"])),
                "phi": np.zeros((0, sizes["phi"])),
                "loss": 0.0
            }
            metrics_col =  np.empty((0, 4))
            
            dataloader = self.dataset.set_loader(mode="predict")
            with tqdm(total=len(dataloader.dataset.files), desc="Processing files", unit="file") as pbar:
                for batch, file_idx, file_completed in dataloader:
                    _, query, _ = batch
                    #query_phi = dataloader.dataset._normalizer.inverse_transform(query.phi[0], "phi").cpu().numpy()
                    #query_theta = dataloader.dataset._normalizer.inverse_transform(query.theta[0], "theta").cpu().numpy()
                    query_phi = dataloader.dataset._normalizer.inverse_transform(query.phi[0],"phi")
                    query_theta = dataloader.dataset._normalizer.inverse_transform(query.theta[0],"theta")

                    # Forward pass
                    with torch.inference_mode():
                        output, targets = self._forward_batch(batch, device)
                    logit = output.get("logits", None)

                    # Update loss
                    collectors["loss"] += (self.criterion(logit, targets) + 
                                         output.get("kl_term", 0.0) + 
                                         output.get("loss", 0.0))

                    # Update predictions
                    if self.model._get_name()== 'ConditionalNeuralProcess':
                        pred_data = logit[0][0].cpu().numpy()
                        collectors["y_err"] = np.concatenate([collectors["y_err"], logit[1][0].cpu().numpy()], axis=0)
                    else:
                        pred_data = torch.sigmoid(logit[0]).cpu().numpy().reshape(-1, 1)
                    
                    # Collect batch data
                    collectors["y_pred"] = np.concatenate([collectors["y_pred"], pred_data], axis=0)
                    collectors["y_true"] = np.concatenate([collectors["y_true"], targets[0].cpu().numpy()], axis=0)
                    collectors["theta"] = np.concatenate([collectors["theta"], query_theta], axis=0)
                    collectors["phi"] = np.concatenate([collectors["phi"], query_phi], axis=0)

                    if file_completed:
                        # Get indices for data validation
                        indices = {k: self.dataset.parameters[k]["selected_indices"] 
                                 for k in ["phi", "theta", "target"]}
                        
                        with h5py.File(self.dataset.files[file_idx], "a") as f:
                            # Load and validate data
                            target_labels = self.dataset.parameters["target"]["selected_labels"]
                            file_data = {
                                "phi": np.array(f[self.dataset.parameters["phi"]["key"]][:,indices["phi"]]),
                                "theta": np.array(f[self.dataset.parameters["theta"]["key"]][indices["theta"]]).reshape(-1, sizes["theta"]),
                                "y_true": np.array(f[self.dataset.parameters["target"]["key"]][:, indices["target"]])
                            }

                            # Validate data consistency
                            data_ok = all(
                                np.allclose(file_data[k], np.asarray(collectors[k], dtype=file_data[k].dtype)[None, :], rtol=1e-5, atol=1e-5)
                                for k in ["theta", "phi", "y_true"]
                            )
                            
                            if not data_ok:
                                print(f"Warning: Data mismatch in file {self.dataset.files[file_idx]}")
                            else:
                                # Set up HDF5 group
                                model_name = self.model.__class__.__name__
                                version = self.dataset.config_file["path_settings"]["version"]
                                group_path = f"RESOLVE_{model_name}_{version}"
                                prefix = group_path
                                to_delete = [name for name in f.keys() if name.startswith(prefix)]
                                for name in to_delete:
                                    del f[name]

                                grp = f.require_group(group_path)
                                
                                # Save predictions and metadata
                                for i,l in enumerate(target_labels):
                                    grp.create_dataset(f'{l}_true', data=collectors["y_true"][:, i], compression="gzip", chunks=True)
                                    grp.create_dataset(f'{l}_pred', data=collectors["y_pred"][:, i], compression="gzip", chunks=True)
                                    if collectors["y_err"].shape[0] > 0:
                                        grp.create_dataset(f'{l}_pred_err', data=collectors["y_err"][:, i], compression="gzip", chunks=True)
                                
                                # Save provenance
                                grp.attrs.update({
                                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "model_name": model_name,
                                    "version": version,
                                    "git_hash": get_git_hash(short=True),
                                    "phi_label": str(self.dataset.parameters["phi"]["selected_labels"]),
                                    "theta_label": str(self.dataset.parameters["theta"]["selected_labels"]),
                                    "target_label": str(self.dataset.parameters["target"]["selected_labels"])
                                })
                            
                                # Save metrics
                                # Save metrics
                                m_v = _compute_metrics(collectors["y_true"], collectors["y_pred"], self.is_binary)
                                m_v["loss"] = float(collectors["loss"])

                                # Correct: update attributes directly from the dictionary
                                grp.attrs.update(m_v)

                        # Log
                        if writer and file_idx % self._report == 0.:
                            for k, v in m_v.items():
                                writer.add_scalar(f"{dataset_name}/{k}", v, file_idx)
                            fig = utils.plot(collectors["y_pred"], collectors["y_true"], it=file_idx)
                            writer.add_figure(f'plot/{dataset_name}', fig, global_step=file_idx)

                        # initialize an empty array with correct number of columns but 0 rows
                        if metrics_col.shape[1] != len(m_v):
                            metrics_col =  np.empty((0, len(m_v)))

                        metrics_col = np.vstack([metrics_col, np.array(list(m_v.values())).reshape(1, -1)])
                        # Reset collectors for next file
                        collectors = {
                                "y_pred": np.zeros((0, sizes["target"])),
                                "y_err": np.zeros((0, sizes["target"])),
                                "y_true": np.zeros((0, sizes["target"])),
                                "theta": np.zeros((0, sizes["theta"])),
                                "phi": np.zeros((0, sizes["phi"])),
                                "loss": 0.0
                            }

                        pbar.update(1)

            for i, (name, _) in enumerate(m_v.items()):
                if dataset_name not in self.metrics:
                    self.metrics[dataset_name] = {}
                # Store both the average and full values
                self.metrics[dataset_name][f'{name}_avg'] = np.nanmean(metrics_col[:, i])
                self.metrics[dataset_name][name] = metrics_col[:, i].tolist()
            
            return {"monitor_avg": np.nanmean(self.metrics[dataset_name].get(monitor, float("nan")))}
                        
