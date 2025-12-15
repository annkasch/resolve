
from multiprocessing import context
import os, time
import gc
from turtle import Turtle
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
in_slurm = "SLURM_JOB_ID" in os.environ
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import h5py
import matplotlib.pyplot as plt
import dataclasses
from ..utilities import utilities as utils
from collections.abc import Mapping, Sequence
from resolve.helpers.losses import bce_with_logits, brier, recon_loss_mse, skip_loss, logit_normal_bernoulli_nll, zero_loss
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F


def validate_coverage(sigma1, sigma2, sigma3, y_data):
        """
        Validate the coverage of the model for 1, 2, and 3 sigma intervals.

        Parameters:
        - y_data (ndarray): True high-fidelity target values for validation.
        - y_hf_pred_samples (ndarray): Posterior predictive samples for high-fidelity predictions.

        Returns:
        - dict: Percentages of validation data within 1, 2, and 3 sigma intervals.
        """

        coverage={}
        counters = {1: 0, 2: 0, 3: 0}
        for ix in range(y_data.shape[0]):

            y_data_tmp = y_data[ix]
            
            #print(sigma1[0][ix], sigma1[1][ix], y_data_tmp)
            # Calculate percentile intervals for the posterior samples

            percentiles = {
                1: (sigma1[0][ix], sigma1[1][ix]),
                2: (sigma2[0][ix], sigma2[1][ix]),
                3: (sigma3[0][ix], sigma3[1][ix]),
            }

            # Count the number of y_data points within each interval
            for sigma in enumerate([1, 2, 3]):
                low, high = percentiles[sigma]
                #print(low, high, y.item())
                in_band = 0
                if low <= y_data_tmp <= high:
                    counters[sigma] += 1
                    in_band = sigma
                if ix < 10: print(low, high, y_data_tmp, in_band)

        # Calculate percentages
        coverage={sigma: (counters[sigma])/y_data.shape[0] * 100. for sigma in [1, 2, 3]}
        return coverage

def get_git_hash(short=True):
    try:
        args = ["git", "rev-parse", "HEAD"]
        if short:
            args.insert(2, "--short")
        return subprocess.check_output(args).decode("utf-8").strip()
    except Exception:
        return "unknown"

def _device() -> torch.device:
    # Prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        print("running on gpu")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("running on mps")
        return torch.device("mps")
    print("running on cpu")
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

def _is_binary_range(target_range: Tuple[float, float]) -> bool:
    lo, hi = target_range
    return lo >= 0.0 and hi <= 1.0

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_binary: bool,
    tol=1e-8
) -> Dict[str, float]:
    """Compact metrics with safe fallbacks for edge cases."""
    metrics: Dict[str, float] = {}
    if is_binary:
        # Probabilities -> labels with 0.5 threshold (customize if needed)
        y_prob = y_pred.reshape(-1)
        y_hat = (y_prob > 0.5).astype(np.int32)
        y_true_i = y_true.reshape(-1)
        if np.any((y_true > tol) & (y_true < 1. - tol)):
            y_true_i = (y_true_i > 0.5)
        y_true_i = y_true_i.astype(np.int32)

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
                fpr, tpr, _ = roc_curve(y_true_i, y_prob)
                metrics["roc_curve"] = [fpr,tpr]
            except Exception:
                metrics["roc_curve"] = float("nan")
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true_i, y_prob))
            except Exception:
                metrics["pr_auc"] = float("nan")
            try:
                precision, recall, _ = precision_recall_curve(y_true_i, y_prob)
                metrics["precision_recall_curve"] = [precision, recall, (y_true_i == 1).mean()]
            except Exception:
                metrics["precision_recall_curve"] = float("nan")
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
        self.nepochs = epochs
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
        
        self.device = _device()
        self.model.to(self.device)

        # AMP policy
        self._use_cuda = (self.device.type == "cuda")
        self._use_bf16 = (self._use_cuda and torch.cuda.is_bf16_supported())
        self._amp_enabled = self._use_cuda  # enable autocast on CUDA; off on MPS/CPU

        # GradScaler only when we might need it (fp16 path)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp_enabled and not self._use_bf16)

        # For logging last epoch metrics
        self.metrics: Dict[str, float] = {}

    def _forward_batch(self, batch: dict, device, train=True, step = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format and forward a single batch. Expects keys consistent with your data pipeline."""
        # unpack the batch:
        context, query, targets = batch

        # move everything to device
        nb = (device.type == "cuda")
        targets = _to_dev(targets, device, non_blocking=nb)
        context = _to_dev(context, device, non_blocking=nb)
        query   = _to_dev(query, device, non_blocking=nb)

        output = self.model(
            query_theta=query.theta, query_phi=query.phi, query_y=targets,
            context_theta=context.theta, context_phi=context.phi, context_y=context.y,
            target_y=targets, query_idx=query.idx, context_idx=context.idx, train=train, step=step
        )

        return output, targets

    def _run_epoch(self, loader, optimizer=None, train: bool = True, desc: str = "train",
                pred_writer=None, file_offsets=None):

        self.model.train(train)
        running_loss = 0.0

        # Keep these only if you need eval metrics; otherwise skip
        y_true_all, y_pred_all, y_score_all, sigma_all = [], [], [], []
        accum_steps = math.ceil(500./loader.dataset.data[loader.dataset.mode]["target"]["batch_size"]) if train==True else 1.

        pbar = tqdm(loader, total=len(loader), desc=desc, leave=True, disable=in_slurm)

        for i, batch in enumerate(pbar):
            with torch.amp.autocast(self.device.type, enabled=self._amp_enabled, dtype=(torch.bfloat16 if self._use_bf16 else torch.float16)):
                output, targets = self._forward_batch(batch, self.device, train=train, step=i+self.epoch*len(loader))

                logit = output.get("logits", None)
                score = output.get("scores", torch.tensor([], device=self.device))
                kl_term = output.get("kl_term", 0.0)
                add_loss = output.get("loss", 0.0)

                _, query, meta = batch  # <- assume meta holds file_id/row_idx (or wherever you store them)
                query_x = torch.cat([query.theta, query.phi], dim=2)
                query_x = _to_dev(query_x, self.device, non_blocking=(self.device.type == "cuda"))

                # Keep loss numerically stable: do loss in fp32 if needed
                # fp32 is safer with custom losses
                if logit[0].dtype != torch.float32:
                    logit32 = [x.float() for x in logit]
                    targets32 = targets.float()
                    qx32 = query_x.float()
                else:
                    logit32, targets32, qx32 = logit, targets, query_x

                loss1 = self.criterion(logit32, targets32, targets_x=qx32)
                loss = loss1 + kl_term + add_loss

            # backward only in training
            if train and self.criterion.base_loss_fn is not skip_loss:
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()


                if (i + 1) % accum_steps == 0:
                    if self.scaler.is_enabled():
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            #if hasattr(self.model, "memory_bank") and self.model.memory_bank is not None:
            #    self.model.memory_bank.ema_update()

            running_loss += float(loss.detach().cpu())

            # === compute prediction tensor ===
            gauss = output.get("Norm", None)
            if gauss is not None:
                pred_t = gauss[0].detach().reshape(-1)
                sigma_t = gauss[1].detach().reshape(-1)
            else:
                sigma_t = None
                if self.criterion.base_loss_fn is bce_with_logits or self.criterion.base_loss_fn is brier:
                    pred_t = torch.sigmoid(logit[0]).detach().reshape(-1)
                else:
                    pred_t = logit[0].detach().reshape(-1)

            # === write predictions during inference ===
            if (not train) and (pred_writer is not None):
                # Pull routing info from batch/meta (adapt these field names!)
                file_id = meta.file_id.detach().cpu().numpy().reshape(-1)
                row_idx = meta.row_idx.detach().cpu().numpy().reshape(-1)

                # compute global indices
                # file_offsets: numpy array/list where file_offsets[f] gives starting index of file f
                global_idx = (np.asarray(file_offsets, dtype=np.int64)[file_id] + row_idx).astype(np.int64)

                pred_np = pred_t.float().detach().cpu().numpy().astype(np.float32)
                if sigma_t is not None:
                    sigma_np = sigma_t.float().detach().cpu().numpy().astype(np.float32)
                    pred_writer.write(global_idx, pred_np, sigma_np)
                else:
                    pred_writer.write(global_idx, pred_np)

            # If you still want metric arrays, keep these; otherwise remove to save RAM
            y_true_all.append(targets.reshape(-1).detach().cpu())
            y_pred_all.append(pred_t.detach().cpu())
            y_score_all.append(score.detach().reshape(-1).cpu())
            if sigma_t is not None:
                sigma_all.append(sigma_t.detach().cpu())

            pbar.set_postfix(loss=f"{running_loss/len(y_true_all):.4f}")

        # return metrics as before (or simplify for inference)
        y_true = torch.cat(y_true_all).float().cpu().numpy() if y_true_all else np.array([])
        y_pred = torch.cat(y_pred_all).float().cpu().numpy() if y_pred_all else np.array([])
        y_score = torch.cat(y_score_all).float().cpu().numpy() if y_score_all else np.array([])
        sigma = torch.cat(sigma_all).float().cpu().numpy() if len(sigma_all) > 0 else np.array([])
        avg_loss = running_loss / max(1, len(y_true_all))
        return avg_loss, y_true, y_pred, y_score, sigma

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        writer=None,
        monitor: str = "pr_auc",  # for binary; for regression we'll silently map to 'rmse'
        mode: str = "max",
        patience: int = 20,
        ckpt_dir: str = "./checkpoints",
        ckpt_name: str = "best.pt",
    ) -> Dict[str, float]:
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt = os.path.join(ckpt_dir, ckpt_name)

        # move memory buffers once; do NOT call .to(device) again in the loop
        #self.model.memory.to(device)

        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.to(self.device)
        
        # rebuild optimizer after model to device
        if self.criterion.base_loss_fn is not skip_loss:
            for s in optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        s[k] = v.to(self.device, non_blocking=(self.device.type=="cuda"))

        # Initialize best score based on objective
        if not self.is_binary and monitor.lower() in {"pr_auc", "roc_auc"}:
            monitor = "rmse"
            mode = "min"

        best_score = -float("inf") if mode == "max" else float("inf")
        best_loss = float("inf")
        best_model_saved = False
        no_improve = 0

        num_epochs = int(self.nepochs*self.dataset.set_loader(0, "train").dataset.data["train"]["meta"]["num_epochs"])

        for epoch in range(self.epoch_start, self.epoch_start + num_epochs):
            # TRAIN
            self.epoch = epoch
            dataloader = self.dataset.set_loader(epoch, "train")

            getattr(self.model, "fit", lambda *args, **kwargs: None)(loader=dataloader, trainer=self)
                
            train_loss, y_true_tr, y_pred_tr, y_score_tr, sigma = self._run_epoch(dataloader, optimizer, train=True, desc=f"train {epoch+1}/{self.epoch_start + num_epochs}")
            m_tr = _compute_metrics(y_true_tr, y_pred_tr, self.is_binary)
            m_tr["loss"] = train_loss
            self.metrics["train"] = m_tr

            # Log
            if writer and epoch % self._report == 0:
                for k, v in m_tr.items(): writer.add_scalar(f"train/{k}", v, epoch+1) if np.isscalar(v) else None

                fig = utils.plot(y_pred_tr.reshape(-1, 1), y_true_tr.reshape(-1, 1), it=epoch+1)
                writer.add_figure(f'plot/score_train', fig, global_step=epoch+1)
                if len(y_score_tr) > 0:
                    fig = utils.plot(y_score_tr.reshape(-1, 1), y_true_tr.reshape(-1, 1), it=epoch+1)
                    writer.add_figure(f'plot/score_tree_train', fig, global_step=epoch+1)
            
            if sigma.size > 0:
                sigma1 = y_pred_tr-sigma, y_pred_tr+sigma
                sigma2 = y_pred_tr-2*sigma, y_pred_tr+2*sigma
                sigma3 = y_pred_tr-3*sigma, y_pred_tr+3*sigma
                validate_coverage_test = validate_coverage(sigma1, sigma2, sigma3, y_true_tr)
                print(f"+- 1sigma: {validate_coverage_test[1]:.2f}; +- 2sigma: {validate_coverage_test[2]:.2f}; +- 3sigma: {validate_coverage_test[3]:.2f}")


            
            # Early stopping / checkpointing
            if "validate":
                score = self.evaluate(writer=writer, dataset_name="validate", monitor=monitor, epoch=epoch)
                improved = (score > best_score) if mode == "max" else (score < best_score)
                loss_tolerance = 0.05  # 5% tolerance on loss for checkpointing
                if improved and best_loss*(1.0 + loss_tolerance) > self.metrics["validate"].get("loss"):
                    best_score = score
                    best_loss = self.metrics["validate"].get("loss")
                    no_improve = 0
                    torch.save({"epoch": epoch, "model_state": self.model.state_dict()}, best_ckpt)
                    best_model_saved = True
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            
            # Memory hygiene
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        if os.path.exists(best_ckpt) and best_model_saved:
            self.load_best_checkpoint(best_ckpt)
        
        self.metrics["best_model"]={"best_score": float(best_score), "monitor": monitor, "mode": mode, "epochs_ran": epoch - self.epoch_start + 1}
        return self.metrics["best_model"]

    def load_best_checkpoint(self, ckpt_path: str):
        """Reload the best model from a checkpoint."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.epoch_start = checkpoint["epoch"] + 1
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        return checkpoint

    def evaluate(
        self,
        writer=None,
        dataset_name="validate",
        monitor: str = "pr_auc",  # for binary; for regression we'll silently map to 'rmse'
        epoch: int = 0,
        fit_temperature: bool = False,
    ) -> Dict[str, float]:
        
        self.epoch = epoch
        #self.model.to(self.device)
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.to(self.device)

        dataloader = self.dataset.set_loader(epoch, dataset_name)
        if dataset_name not in dataloader.dataset.data: 
            return
        with torch.inference_mode():
            loss, y_true_v, y_pred_v, y_score_v, sigma = self._run_epoch(dataloader, optimizer=None, train=False, desc=f"{dataset_name} {epoch+1}")
        
        '''
        # Optionally FIT TEMPERATURE on validation set
        T_used = None
        if self.is_binary:
            # Decide what temperature to use / fit

            if fit_temperature:
                # Reconstruct logits from probabilities
                eps = 1e-6
                probs = torch.as_tensor(y_pred_v, device=self.device, dtype=torch.float32)
                probs = probs.clamp(eps, 1.0 - eps)
                logits = torch.logit(probs, eps=eps)
                targets = torch.as_tensor(y_true_v, device=self.device, dtype=torch.float32)

                # Scalar log_T parameter (T > 0 via exp)
                log_T = torch.zeros(1, device=self.device, requires_grad=True)
                optimizer_T = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)

                def closure():
                    optimizer_T.zero_grad()
                    T = torch.exp(log_T)
                    logits_T = logits / T
                    # plain BCE for calibration, regardless of training loss
                    calib_loss = F.binary_cross_entropy_with_logits(logits_T, targets)
                    calib_loss.backward()
                    return calib_loss.detach()

                # Enable grads just for this fit
                with torch.enable_grad():
                    optimizer_T.step(closure)

                T_used = torch.exp(log_T).item()
                # Store on the object for later
                self.model.temperature.data.fill_(T_used)
                
            else:
                # If we already have a learned temperature, use it
                if hasattr(self.model, "temperature"):
                    T_used = float(self.model.temperature.item())
                else:
                    T_used = None
            
            # Apply temperature (if any) and convert to probabilities
            if T_used is not None:
                eps = 1e-6
                probs = torch.as_tensor(y_pred_v, device=self.device, dtype=torch.float32)
                probs = probs.clamp(eps, 1 - eps)
                logits = torch.log(probs / (1.0 - probs))  # logit(p)
                logits_T = logits / T_used
                probs = torch.sigmoid(logits_T).float().cpu().numpy()        # calibrated probs
            else:
                probs = y_pred_v                       # original probs
        else:
            # Non-binary mode: keep predictions as-is
            probs = y_pred_v
        '''
        probs = y_pred_v

        # Compute metrics using calibrated probabilities
        m_v = _compute_metrics(y_true_v, y_pred_v, self.is_binary)
        m_v["loss"] = loss
        self.metrics[dataset_name] = m_v
        if sigma.size > 0:
            sigma1 = y_pred_v-sigma, y_pred_v+sigma
            sigma2 = y_pred_v-2*sigma, y_pred_v+2*sigma
            sigma3 = y_pred_v-3*sigma, y_pred_v+3*sigma
            validate_coverage_test = validate_coverage(sigma1, sigma2, sigma3, y_true_v)
            print(f"+- 1sigma: {validate_coverage_test[1]:.2f}; +- 2sigma: {validate_coverage_test[2]:.2f}; +- 3sigma: {validate_coverage_test[3]:.2f}")

        # Logging / plots
        if writer and epoch % self._report == 0:
            for k, v in m_v.items():
                if np.isscalar(v):
                    writer.add_scalar(f"{dataset_name}/{k}", v, epoch)

            # Main score plot: use calibrated probs
            fig = utils.plot(probs.reshape(-1, 1), y_true_v.reshape(-1, 1), it=epoch+1)
            writer.add_figure(f'plot/score_{dataset_name}', fig, global_step=epoch+1)

            # Tree score plot
            if len(y_score_v) > 0:
                fig = utils.plot(y_score_v.reshape(-1, 1), y_true_v.reshape(-1, 1), it=epoch+1)
                writer.add_figure(f'plot/score_tree_{dataset_name}', fig, global_step=epoch+1)

            if "precision_recall_curve" in m_v and isinstance(m_v["precision_recall_curve"], list):
                fig = plt.figure()
                plt.plot(m_v["precision_recall_curve"][0], m_v["precision_recall_curve"][1])
                plt.xlabel("Signal Efficiency (Recall)")
                plt.ylabel("Precision")
                writer.add_figure(f'plot/prec_recall_{dataset_name}', fig, global_step=epoch+1)

            if "roc_curve" in m_v and isinstance(m_v["roc_curve"], list):
                fig = plt.figure()
                plt.plot(m_v["roc_curve"][1], 1 - m_v["roc_curve"][0])
                plt.xlabel("Signal Efficiency")
                plt.ylabel("Background Efficiency")
                writer.add_figure(f'plot/roc_curve_{dataset_name}', fig, global_step=epoch+1)


        score = m_v.get(monitor.lower(), m_v.get(monitor, float("nan")))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return score

    @torch.inference_mode()
    def predict(self, dataset_name="inference", epoch=0, out_path="out/preds.dat",
                      n_total=None, file_offsets=None, with_sigma=False):
        self.epoch = epoch
        dataloader = self.dataset.set_loader(epoch, dataset_name)

        if n_total is None:
            n_total = self.dataset.n_total_samples  # adapt to your dataset
        if file_offsets is None:
            file_offsets = self.dataset.file_offsets  # prefix sums by file_id

        writer = MemmapPredWriter(out_path, n_total=n_total, with_sigma=with_sigma)

        with torch.inference_mode():
            self._run_epoch(dataloader, optimizer=None, train=False, desc=f"{dataset_name} {epoch+1}",
                            pred_writer=writer, file_offsets=file_offsets)

        writer.close()
        return out_path

    def warm_up(
        self,
        target_pos_frac: Union[float, Sequence[float]],
        optimizer: torch.optim.Optimizer,
        writer=None,
        monitor: str = "pr_auc",
        mode: str = "max",               # 👈 now mandatory input
        patience: int = 15,
        ckpt_dir: str = "./checkpoints",
        ckpt_name: str = "best.pt",
        num_data_pass_per_phase: Optional[int] = 1,
    ) -> Dict[str, float]:
        """
        Warm-up training with staged positive-fraction schedule.
        """
        num_epochs = self.nepochs 
        self.nepochs = num_data_pass_per_phase
        dataloader = self.dataset.set_loader(0, "train")

        if self.model._get_name() == 'TreeConditionedCNP' and self.model.tree._fitted == False:
                self.model.fit(loader=dataloader)
                self.model.tree.enable_leaf_cache(dataloader.dataset.num_samples())
        self.dataset.dataset = None
        counter = 0 
        for ratio in target_pos_frac:
            
            self.dataset.config_file["model_settings"]["train"]["dataset"]["positive_ratio_train"]=ratio
            self.dataset.set_dataset()
            print(f"----- Initializing warm-up phase — positives set to {self.dataset.dataset.data["train"]["meta"]["pos_frac"]:.2f} of the batch.----")
            self.fit(optimizer=optimizer, patience = patience, writer=writer, ckpt_dir=ckpt_dir, ckpt_name=ckpt_name,
            monitor=monitor, mode=mode)
            counter += self.nepochs
            self.epoch_start = counter
            
        
        self.dataset.config_file["model_settings"]["train"]["dataset"]["positive_ratio_train"] = None
        self.dataset.dataset = None
        self.epoch_start = counter
        self.nepochs = num_epochs
        print(f"----- End of warm up -----")


class MemmapPredWriter:
        def __init__(self, path: str, n_total: int, with_sigma: bool = False, dtype=np.float32):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.p_path = path
            self.p = np.memmap(path, mode="w+", dtype=dtype, shape=(n_total,))
            self.with_sigma = with_sigma
            self.s = None
            if with_sigma:
                self.s_path = path.replace(".dat", "_sigma.dat")
                self.s = np.memmap(self.s_path, mode="w+", dtype=dtype, shape=(n_total,))

        def write(self, global_idx: np.ndarray, pred: np.ndarray, sigma: np.ndarray | None = None):
            self.p[global_idx] = pred
            if self.with_sigma and sigma is not None:
                self.s[global_idx] = sigma

        def close(self):
            self.p.flush()
            if self.s is not None:
                self.s.flush()