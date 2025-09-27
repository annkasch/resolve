import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import get_worker_info
import torch
from .losses import AsymmetricFocalWithFPPenalty
from sklearn.metrics import precision_recall_curve, auc
import os
import zarr, gc
from numcodecs import Blosc
from typing import Optional, Tuple
from .zarr_helper import ZarrPredWriter


class Trainer():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.target_range = self.dataset.config_file["simulation_settings"]["target_range"]
        self.is_binary = self.target_range[0] >= 0 and self.target_range[1] <= 1
        self.training_epochs = 10
        self.epoch_start = 0
        self.criterion = AsymmetricFocalWithFPPenalty()
        

    def fit(self, optimizer, writer, dataset_val, debug = "off",
        ckpt_dir="./checkpoints", ckpt_name="best.pt",
        monitor="PR-AUC", mode="max", patience=10):
        
        os.makedirs(ckpt_dir, exist_ok=True)
        best_ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        device = torch.device("cpu")
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        best_score = float("-inf") if mode == "max" else float("inf")
        epochs_no_improve = 0

        if debug == "profiling":
            # --- add profiler here ---
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_stack=False
            )
            profiler.__enter__()
        # -------------------------
        
        max_grad_norm = None  # turn off unless you actually need it for stability
        for it_epoch in range(self.epoch_start, self.epoch_start + self.training_epochs):
            self.dataset.set_loader()
            dataloader = self.dataset.dataloader
            running_loss = 0.0
            n_batches = 0

            y_pred_list = []
            y_true_list = []

            # If format_batch_for_cnp is heavy Python work, consider moving it into your Dataset __getitem__
            # so each sample is already in the right shape; then your collate_fn can stack efficiently.
            for b, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / self.dataset._batch_size), desc=f"Training Epoch {it_epoch+1}/{self.training_epochs}"):
                batch_formatted = self.dataset.format_batch_for_cnp(
                    batch, self.dataset.config_file["cnp_settings"]["context_is_subset"]
                )
            
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                logit, kl_term = self.model(
                    batch_formatted.context.theta,
                    batch_formatted.context.phi,
                    batch_formatted.context.y,
                    batch_formatted.query.theta,
                    batch_formatted.query.phi
                )

                target = batch_formatted.target_y.to(device=device, dtype=torch.float32)
                loss = self.criterion(logit, target) + kl_term

                # Backward + step
                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                optimizer.step()

                with torch.no_grad():
                    y_pred_list.append(torch.sigmoid(logit).detach().cpu().flatten())
                    y_true_list.append(target.detach().cpu().flatten())

                running_loss += float(loss)
                n_batches += 1

                # --- stop after a few batches just for profiling ---
                #if b == 100 and debug == "profiling":   # profile first 10 batches
                #    break
            # end inner loop
            self.dataset.close_loader()
            if debug == "profiling":
                # --- exit profiler and print report ---
                profiler.__exit__(None, None, None)
                print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
                return  # stop after profiling one epoch
            
            # Concatenate; convert to numpy
            y_pred = torch.cat(y_pred_list, dim=0).numpy()
            y_true = torch.cat(y_true_list, dim=0).numpy()

            epoch_loss = running_loss / max(1, n_batches)
            self.metrics_train = self.log_metrics(writer, y_true, y_pred, epoch_loss, it_epoch, self.is_binary, leg="Train")

            # Make sure your evaluate() uses torch.inference_mode() and avoids per-batch numpy conversions too
            y_pred_val, y_true_val, loss_val = self.evaluate(dataset_val)
            self.metrics_val = self.log_metrics(writer, y_true_val, y_pred_val, loss_val, it_epoch, self.is_binary, leg="Val")

            print(f"[Epoch {it_epoch+1} Summary]")
            print(f'loss (train)   = {self.metrics_train["Loss"]:.5f} loss (val)    = {self.metrics_val["Loss"]:.5f}')
            print(f'PR-AUC (train) = {self.metrics_train["PR-AUC"]:.5f} PR-AUC (val) = {self.metrics_val["PR-AUC"]:.5f}')

            # ---- checkpointing on validation metric ----
            current = self.metrics_val.get(monitor, None)
            if current is None:
                raise ValueError(f"Metric '{monitor}' not found in self.metrics_val keys: {list(self.metrics_val.keys())}")

            is_better = (current > best_score) if mode == "max" else (current < best_score)

            if is_better:
                best_score = current
                epochs_no_improve = 0
                torch.save({
                    "epoch": it_epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "monitor": monitor,
                    "mode": mode,
                    "config": {
                        "weight_signal_rate": float(self.criterion.prior_pos),
                        "gamma_pos": self.criterion.gamma_pos, "gamma_neg": self.criterion.gamma_neg, "alpha_pos": self.criterion.alpha_pos, "alpha_neg": self.criterion.alpha_neg,
                        "lambda_fp": self.criterion.lambda_fp, "tau_fp": self.criterion.tau_fp
                    }
                }, best_ckpt_path)
                print(f"Saved new best to {best_ckpt_path} ({monitor}={best_score:.6f})")
            else:
                epochs_no_improve += 1
                if (patience is not None) and (epochs_no_improve >= patience):
                    print(f"Early stopping at epoch {it_epoch+1}: no improvement in {patience} epochs.")
                    break

        # ---- at the end: restore best weights ----
        if os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"Restored best model from {best_ckpt_path} ({ckpt['monitor']}={ckpt['best_score']:.6f})")
        else:
            print("No checkpoint found; leaving model at last epoch weights.")
            
            
    def evaluate(self, dataset_eval):
        dataset_eval.set_loader()
        dataloader = dataset_eval.dataloader
        
        device = torch.device("cpu")
        self.model.to(device)
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        y_pred_list = []
        y_true_list = []
        
        with torch.no_grad():

            for b, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / dataset_eval._batch_size), desc=f"Evaluation"):
                batch_formatted = dataset_eval.format_batch_for_cnp(
                    batch, self.dataset.config_file["cnp_settings"]["context_is_subset"]
                )
                # Forward pass
                logit, kl_term = self.model(
                    batch_formatted.context.theta,
                    batch_formatted.context.phi,
                    batch_formatted.context.y,
                    batch_formatted.query.theta,
                    batch_formatted.query.phi
                )

                target = batch_formatted.target_y.to(device=device, dtype=torch.float32)
                loss = self.criterion(logit, target) + kl_term

                y_pred_list.append(torch.sigmoid(logit).detach().cpu().flatten())
                y_true_list.append(target.detach().cpu().flatten())

                running_loss += float(loss)
                n_batches += 1

        # Combine predictions and targets
        y_pred = torch.cat(y_pred_list, dim=0).numpy()
        y_true = torch.cat(y_true_list, dim=0).numpy()

        epoch_loss = running_loss / max(1, n_batches)
        dataset_eval.close_loader()

        return y_pred, y_true, epoch_loss

    def predict(
        self,
        dataset_predict,
        writer: Optional[ZarrPredWriter] = None,
        out_store: str = "pred_store.zarr",
        dtype=np.float32,
        mu_chunks: int = 262_144,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns small aggregates in RAM:
        theta         : (n_files, theta_size)
        counts        : (n_files, 1)
        mu_mean       : (n_files, 1)
        sigma_mean    : (n_files, 1)
        y_mean        : (n_files, 1) if scalar y, else 0
        Persists per-query data to Zarr via ZarrPredWriter.
        """
        self.model.eval()

        # basic dims
        theta_size = len(dataset_predict._names_theta)
        q_theta_dim = len(dataset_predict._names_theta)
        q_phi_dim = len(dataset_predict._names_phi)
        y_dim = len(dataset_predict._names_target)  # 1 for scalar targets

        # writer
        _writer = writer or ZarrPredWriter(out_store, theta_size=theta_size, dtype=dtype, mu_chunks=mu_chunks)

        # aggregates to return
        mu_mean, sigma_mean, y_mean, theta_rows, counts = [], [], [], [], []

        # open dataloader
        dataset_predict.set_loader()
        dataloader = dataset_predict.dataloader
        pbar = tqdm(total=len(dataloader.dataset.files), desc="Processing files", unit="file")

        with torch.no_grad():
            for batch, worker_id, file_completed in dataloader:
                # format batch and run model
                bf = dataset_predict.format_batch_for_cnp(
                    batch,
                    self.dataset.config_file["cnp_settings"]["context_is_subset"]
                )
                # model forward
                logit, _ = self.model(
                    bf.context.theta, bf.context.phi, bf.context.y,
                    bf.query.theta, bf.query.phi
                )
                mu0_t = torch.sigmoid(logit)[0].detach().cpu().float()            # (N,)
                y0_t  = bf.target_y[0].detach().cpu().float()                     # (N,) or (N, y_dim)

                # unnormalize query features to RAW for storage
                theta_mean_t = dataset_predict.feature_mean.cpu()[:, :theta_size]
                theta_std_t  = dataset_predict.feature_std.cpu()[:, :theta_size]
                phi_mean_t   = dataset_predict.feature_mean.cpu()[:, theta_size:]
                phi_std_t    = dataset_predict.feature_std.cpu()[:, theta_size:]

                q_theta_raw_t = bf.query.theta * (theta_std_t + 1e-6) + theta_mean_t
                q_phi_raw_t   = bf.query.phi * (phi_std_t + 1e-6) + phi_mean_t

                mu0 = mu0_t.numpy().astype(dtype, copy=False).ravel()
                q_theta_raw = q_theta_raw_t[0].detach().cpu().numpy().astype(dtype, copy=False)  # (N, q_theta_dim)
                q_phi_raw   = q_phi_raw_t[0].detach().cpu().numpy().astype(dtype, copy=False)    # (N, q_phi_dim)
                y0_raw      = y0_t.numpy()

                # ensure arrays exist for this worker
                W = _writer.ensure_worker(int(worker_id), q_theta_dim=q_theta_dim, q_phi_dim=q_phi_dim, y_dim=y_dim)

                # append batch to store + update streaming stats
                _writer.append_batch(int(worker_id), mu=mu0, q_theta=q_theta_raw, q_phi=q_phi_raw, y_true=y0_raw, y_dim=y_dim)

                # one per-file theta row (context) for meta and return arrays
                # take the first query's theta (already unnormalized)
                theta_row = q_theta_raw[0].astype(dtype, copy=False)

                if file_completed:
                    mu_avg, sigma_avg, y_avg, n = _writer.finalize_file(int(worker_id), theta_row=theta_row)
                    mu_mean.append(mu_avg)
                    sigma_mean.append(sigma_avg)
                    y_mean.append(y_avg if y_dim == 1 else 0.0)
                    theta_rows.append(theta_row)
                    counts.append(n)
                    pbar.update(1)

                # housekeeping for very large runs
                if W["mu"].shape[0] and (W["mu"].shape[0] % (mu_chunks * 8) == 0):
                    gc.collect()

        dataset_predict.close_loader()

        # return small aggregates
        theta_arr = np.asarray(theta_rows, dtype=dtype)
        counts_arr = np.asarray(counts, dtype=np.int64).reshape(-1, 1)
        mu_mean_arr = np.asarray(mu_mean, dtype=dtype).reshape(-1, 1)
        sigma_mean_arr = np.asarray(sigma_mean, dtype=dtype).reshape(-1, 1)
        y_mean_arr = np.asarray(y_mean, dtype=dtype).reshape(-1, 1)
        return theta_arr, counts_arr, mu_mean_arr, sigma_mean_arr, y_mean_arr

    """
    def predict(self, dataset_predict, writer=None):

            #Run the model in prediction mode over the given dataset.
            #Returns:
            #    - mu_all: np.ndarray of predicted means
            #    - sigma_all: np.ndarray of predicted standard deviations
            #    - target_y_all: np.ndarray of ground truth targets
            #    - features_all: np.ndarray of unnormalized input features

            self.model.eval()
            theta_size = len(dataset_predict.config_file["simulation_settings"]["theta_headers"])
            num_workers = dataset_predict.config_file["cnp_settings"]["dataloader_number_of_workers"]

            mu_mean=[]
            sigma_mean=[]
            y_mean=[]
            theta=[]
            mu_mean_tmp = [[] for _ in range(num_workers)]
            sigma_mean_tmp = [[] for _ in range(num_workers)]
            y_mean_tmp = [[] for _ in range(num_workers)]
            counts = []

            dataset_predict.set_loader()
            dataloader = dataset_predict.dataloader

            # Progress bar for files
            pbar = tqdm(total=len(dataloader.dataset.files), desc="Processing files", unit="file")

            with torch.no_grad():
                for batch, worker_id, file_completed in dataloader:
                    batch_formatted = dataset_predict.format_batch_for_cnp(
                            batch, self.dataset.config_file["cnp_settings"]["context_is_subset"]
                    )
                    # Forward pass
                    logit, _ = self.model(
                        batch_formatted.context.theta,
                        batch_formatted.context.phi,
                        batch_formatted.context.y,
                        batch_formatted.query.theta,
                        batch_formatted.query.phi
                    )
                    mu = torch.sigmoid(logit)
                    # Denormalize features
                    theta_norm = batch_formatted.context.theta.cpu()
                    theta_mean = dataset_predict.feature_mean.cpu()[:,:theta_size]
                    theta_std = dataset_predict.feature_std.cpu()[:,:theta_size]
                    theta_unnorm = theta_norm * (theta_std+1.e-6) + theta_mean  # element-wise

                    mu_mean_tmp[worker_id].extend(mu[0].cpu().numpy())
                    sigma_mean_tmp[worker_id].extend(mu[0].cpu().numpy())
                    y_mean_tmp[worker_id].extend(batch_formatted.target_y[0].cpu().numpy())

                    if file_completed:
                        mu_avg = np.mean(mu_mean_tmp[worker_id])
                        sigma_avg = 0.
                        
                        mu_mean.append(mu_avg)
                        sigma_mean.append(sigma_avg)
                        y_mean.append(np.mean(y_mean_tmp[worker_id]))
                        theta.append(theta_unnorm[0].cpu().numpy()[0][:theta_size])
                        counts.append(len(mu_mean_tmp[worker_id]))
                        
                        if writer != None:
                            fig = self.plot(np.array(mu_mean_tmp[worker_id]), np.array(y_mean_tmp[worker_id]), r"$\bar{y}_{CNP}$="+f"{mu_mean[-1]:.3f}", self.target_range, len(mu_mean))
                            writer.add_figure(f'Plot/Predition', fig, global_step=len(mu_mean))
                        
                        mu_mean_tmp[worker_id]=[]
                        y_mean_tmp[worker_id]=[]
                        sigma_mean_tmp[worker_id]=[]
                        
                        pbar.update(1)  # move progress bar by one file
            dataset_predict.close_loader()
            return np.array(theta), np.array(counts).reshape(-1,1), np.array(mu_mean).reshape(-1,1), np.array(sigma_mean).reshape(-1,1), np.array(y_mean).reshape(-1,1)
    """
    def log_metrics(self, writer, y_true, y_pred, loss, it_step, is_binary, leg="Train"):
        metrics = {
                "Loss": loss,
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
            }
        writer.add_scalar(f'Loss/{leg}', loss, it_step)
        writer.add_scalar(f'Metrics/Mae/{leg}', metrics["MAE"], it_step)
        writer.add_scalar(f'Metrics/Mse/{leg}', metrics["MSE"], it_step)
        writer.add_scalar(f'Metrics/R2/{leg}', metrics["R2"], it_step)

        if is_binary:
            y_pred_bin = (y_pred > 0.5).astype(int)
            y_true_bin = y_true.astype(int)
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
            metrics.update({
                    "PR-AUC": auc(recall, precision),
                    "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
                    "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
                    "Recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
                    "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_true_bin, y_pred_bin),
                    "Average_Precision": average_precision_score(y_true_bin, y_pred_bin)
                })
            
            writer.add_scalar(f'Metrics/PR-AUC/{leg}', metrics["PR-AUC"], it_step)
            writer.add_scalar(f'Metrics/Accuracy/{leg}', metrics["Accuracy"], it_step)
            writer.add_scalar(f'Metrics/Precision/{leg}', metrics["Precision"], it_step)
            writer.add_scalar(f'Metrics/Recall/{leg}', metrics["Recall"], it_step)
            writer.add_scalar(f'Metrics/F1/{leg}', metrics["F1"], it_step)
            writer.add_scalar(f'Metrics/ROC_AUC/{leg}', metrics["ROC_AUC"], it_step)
            writer.add_scalar(f'Metrics/Average_Precision/{leg}', metrics["Average_Precision"], it_step)
        
        fig = self.plot(y_pred, y_true, f'Loss {loss:.4f}', self.target_range, it_step)
        writer.add_figure(f'Plot/{leg}', fig, global_step=it_step)

        return metrics
    
    def get_final_metrics(self):
        # Extract or define final metrics manually if not tracked
        version=self.dataset.config_file["path_settings"]["version"]
        path_out=f'{self.dataset.config_file["path_settings"]["path_out_cnp"]}/{version}'

        final_summary = {
            "Final Hyperparameters": {
                "Encoder Sizes": self.dataset.config_file["cnp_settings"]["encoder_hidden_layers"],
                "Decoder Sizes": self.dataset.config_file["cnp_settings"]["decoder_hidden_layers"],
                "Representation Size": self.dataset.config_file["cnp_settings"]["representation_size"],
                "Learning Rate": self.dataset.config_file["cnp_settings"]["learning_rate"],
                "Batch Size": self.dataset.batch_size,
                "Files per Batch": self.dataset.files_per_batch,
                "Training Epochs": self.dataset.config_file["cnp_settings"]["trainings_epochs"],
                "Binary Task": self.is_binary,
                "Data Augmentation": self.dataset.config_file["cnp_settings"]["use_data_augmentation"],
                "Context is Subset": self.dataset.config_file["cnp_settings"]["context_is_subset"],
            },
            "Output Path": {path_out},
            "Model Checkpoint": f'{path_out}/cnp_{version}_model.pth',
            "TensorBoard Logs": f'{path_out}/cnp_{version}_tensorboard_logs',
            "Final Training Metrics": self.metrics_train,
            "Final Evaluation Metrics": self.metrics_eval
        }
        return final_summary
    
    def plot(self, prediction_y_training, target_y_training, sub_title="", target_range=[0, 1], it=None):
        # Handle both 1D and 2D inputs
        if prediction_y_training.ndim == 1:
            prediction_y_training = prediction_y_training[:, np.newaxis]
        if target_y_training.ndim == 1:
            target_y_training = target_y_training[:, np.newaxis]

        n_outputs = prediction_y_training.shape[1]

        # Create mask for signal (assumes binary classification where signal ~1)
        if target_range[1] == 1:
            mask_signal = target_y_training > 0.5
        else:
            mask_signal = np.full_like(target_y_training, False, dtype=bool)

        mask_bkg = ~mask_signal

        # Split predictions and targets by signal/background
        prediction_signal_training = prediction_y_training[mask_signal].reshape(-1, n_outputs)
        target_signal_training = target_y_training[mask_signal].reshape(-1, n_outputs)
        prediction_bkg_training = prediction_y_training[mask_bkg].reshape(-1, n_outputs)
        target_bkg_training = target_y_training[mask_bkg].reshape(-1, n_outputs)

        fig, ax = plt.subplots(ncols=n_outputs, figsize=(5 * n_outputs, 4))

        if n_outputs == 1:
            ax = [ax]

        if it is not None:
            fig.suptitle(f'Iteration {it}', fontsize=10)

        bins = 100

        for k in range(n_outputs):
            has_signal = target_signal_training.shape[0] > 0

            if has_signal:
                ax[k].hist(target_signal_training[:, k], range=target_range, bins=bins, color='orangered', alpha=1.0, label='label (signal)')

            ax[k].hist(target_bkg_training[:, k], range=target_range, bins=bins, color=(3/255, 37/255, 46/255), alpha=0.8, label='label (bkg)')
            ax[k].hist(prediction_bkg_training[:, k], range=target_range, bins=bins, color=(113/255, 150/255, 159/255), alpha=0.8, label='network (bkg)')

            if has_signal:
                ax[k].hist(prediction_signal_training[:, k], range=target_range, bins=bins, color='coral', alpha=0.8, label='network (signal)')

            ax[k].set_yscale('log')
            ax[k].set_ylim(1e-1,1e6)
            ax[k].set_ylabel("Count")
            ax[k].set_xlabel(r'$y_{CNP}$')
            ax[k].set_title(sub_title, fontsize=10)

            fig.subplots_adjust(bottom=0.3, wspace=0.33)

            if has_signal:
                ax[k].legend(
                    labels=['label (signal)', 'label (bkg)', 'network (bkg)', 'network (signal)'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=2,
                    frameon=False
                )
            else:
                ax[k].legend(
                    labels=['label', 'network'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.2),
                    ncol=2,
                    frameon=False
                )

        return fig