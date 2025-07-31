import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import math


class Trainer():
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.target_range = self.dataset.config_file["simulation_settings"]["target_range"]
        self.is_binary = self.target_range[0] >= 0 and self.target_range[1] <= 1
        self.training_epochs = int(self.dataset.config_file["cnp_settings"]["training"]["phase1"]["training_epochs"])
        self.weight_fp=0.
        self.weight_tp=0.
        self.epoch_start = 0

    def log_metrics(self, writer, y_true, y_pred, loss, it_step, is_binary, leg="Train"):
        metrics = {
                "LogProb": loss,
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
            }
        writer.add_scalar(f'Loss/Logprob/{leg}', loss, it_step)
        writer.add_scalar(f'Metrics/Mae/{leg}', metrics["MAE"], it_step)
        writer.add_scalar(f'Metrics/Mse/{leg}', metrics["MSE"], it_step)
        writer.add_scalar(f'Metrics/R2/{leg}', metrics["R2"], it_step)

        if is_binary:
            y_pred_bin = (y_pred > 0.5).astype(int)
            y_true_bin = y_true.astype(int)
            metrics.update({
                    "Accuracy": accuracy_score(y_true_bin, y_pred_bin),
                    "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
                    "Recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
                    "F1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
                    "ROC_AUC": roc_auc_score(y_true_bin, y_pred_bin),
                    "Average_Precision": average_precision_score(y_true_bin, y_pred_bin)
                })
            writer.add_scalar(f'Metrics/Accuracy/{leg}', metrics["Accuracy"], it_step)
            writer.add_scalar(f'Metrics/Precision/{leg}', metrics["Precision"], it_step)
            writer.add_scalar(f'Metrics/Recall/{leg}', metrics["Recall"], it_step)
            writer.add_scalar(f'Metrics/F1/{leg}', metrics["F1"], it_step)
            writer.add_scalar(f'Metrics/ROC_AUC/{leg}', metrics["ROC_AUC"], it_step)
            writer.add_scalar(f'Metrics/Average_Precision/{leg}', metrics["Average_Precision"], it_step)
        
        fig = self.plot(y_pred, y_true, loss, self.target_range, it_step)
        writer.add_figure(f'Plot/{leg}', fig, global_step=it_step)

        return metrics

    def plot(self, prediction_y_training, target_y_training, loss_training, target_range=[0, 1], it=None):
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
            fig.suptitle(f'Training Iteration {it}', fontsize=10)

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
            ax[k].set_ylabel("Count")
            ax[k].set_xlabel(r'$y_{CNP}$')
            ax[k].set_title(f'Loss {loss_training:.4f}', fontsize=10)

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
    
    def evaluate(self, dataset_eval):
        self.model.eval()
        bce = nn.BCELoss()

        y_preds = []
        y_trues = []
        total_logprob_loss = 0.0
        total_bce_loss = 0.0
        total_samples = 0

        dataloader = dataset_eval.dataloader
        with torch.no_grad():
            for b, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / dataset_eval._batch_size), desc=f"Evaluation"):
                batch_formatted = dataset_eval.format_batch_for_cnp(
                    batch, self.dataset.config_file["cnp_settings"]["context_is_subset"]
                )
                log_prob, mu, _ = self.model(batch_formatted.query, batch_formatted.target_y, self.is_binary)

                y_pred_batch = mu[0].cpu().numpy().flatten()
                y_true_batch = batch_formatted.target_y[0].cpu().numpy().flatten()

                y_preds.append(y_pred_batch)
                y_trues.append(y_true_batch)

                batch_size = len(y_true_batch)
                total_logprob_loss += (-log_prob.mean().item()) * batch_size
                if self.is_binary:
                    total_bce_loss += bce(mu, batch_formatted.target_y).item() * batch_size
                total_samples += batch_size

        # Combine predictions and targets
        y_pred = np.concatenate(y_preds)
        y_true = np.concatenate(y_trues)

        # Compute final averaged losses
        avg_logprob_loss = total_logprob_loss / total_samples
        avg_bce_loss = total_bce_loss / total_samples if self.is_binary else None

        return y_pred, y_true, avg_logprob_loss, avg_bce_loss
    
    def fit(self, optimizer, writer, dataset_val):
        bce = nn.BCELoss()
        dataloader = self.dataset.dataloader
        for it_epoch in range(self.epoch_start, self.epoch_start+self.training_epochs):
            y_preds = []
            y_trues = []
            total_logprob_loss = 0.0
            total_bce_loss = 0.0
            total_samples = 0
            for b, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader) / self.dataset._batch_size), desc=f"Training Epoch {it_epoch+1}/{self.training_epochs}"):
                #it_step = it_epoch * math.ceil(len(dataloader) / self.dataset._batch_size) + b
                batch_formatted = self.dataset.format_batch_for_cnp(batch, self.dataset.config_file["cnp_settings"]["context_is_subset"])

                log_prob, mu, _ = self.model(batch_formatted.query, batch_formatted.target_y, self.is_binary)
                #writer.add_scalar('Loss/Logprob/Train', loss.item(), it_step)

                # Define the loss
                loss = -log_prob.mean()
                loss += self.penalty_loss_fp(mu[0], batch_formatted.target_y[0], self.weight_fp, self.weight_tp)
                loss.backward()
                # Perform gradient descent to update parameters
                optimizer.step()
            
                # reset gradient to 0 on all parameters
                optimizer.zero_grad()
                
                if self.is_binary:
                    loss_bce = bce(mu, batch_formatted.target_y)
                else:
                    loss_bce=-1
                
                y_pred_batch = mu[0].detach().cpu().numpy().flatten()
                y_true_batch = batch_formatted.target_y[0].detach().cpu().numpy().flatten()

                y_preds.append(y_pred_batch)
                y_trues.append(y_true_batch)

                batch_size = len(y_true_batch)
                total_logprob_loss += (-log_prob.mean().item()) * batch_size
                if self.is_binary:
                    total_bce_loss += bce(mu, batch_formatted.target_y).item() * batch_size
                total_samples += batch_size

            # Combine predictions and targets
            y_pred = np.concatenate(y_preds)
            y_true = np.concatenate(y_trues)

            # Compute final averaged losses
            avg_logprob_loss = total_logprob_loss / total_samples
            avg_bce_loss = total_bce_loss / total_samples if self.is_binary else None

            self.metrics_train = self.log_metrics(writer, y_true, y_pred, avg_logprob_loss, it_epoch, self.is_binary, leg="Train")   
            self.metrics_train['BCE'] = avg_bce_loss if self.is_binary else None 
            y_pred_val,  y_true_val, loss_logprob, loss_bce = self.evaluate(dataset_val)
            self.metrics_eval = self.log_metrics(writer, y_true_val, y_pred_val, loss_logprob, it_epoch, self.is_binary, leg="Eval")
            self.metrics_eval['BCE'] = loss_bce if self.is_binary else None
            writer.add_scalar('Loss/BCE/Eval', loss_bce, it_epoch)

    def penalty_loss_fp(self, y_pred, y_true, weight_fp=1.0, weight_tp=0.1):
        """
        Penalize background events misclassified as signal (FP),
        and reward correct signal classifications (TP).
        y_true: ground truth labels (0 for background, 1 for signal)
        y_pred: predicted probabilities ∈ [0, 1] (after sigmoid)
        """
        # False positives: background misclassified as signal
        mask_fp = (y_true < 0.5) & (y_pred >= 0.5)
        penalty_fp = (y_pred - 0.5) * mask_fp.float()

        # True positives: signal correctly identified
        mask_tp = (y_true >= 0.5) & (y_pred >= 0.5)
        reward_tp = (y_pred - 0.5) * mask_tp.float()

        return weight_fp * penalty_fp.mean() - weight_tp * reward_tp.mean()

    def focal_loss_soft(self,y_pred, y_true, alpha=0.99, gamma=2.0, eps=1e-8, reduction='mean'):
        """
        Focal loss for soft targets (e.g. from Mixup).
        y_pred: Tensor of shape (batch,) or (batch, 1), after sigmoid
        y_true: Tensor of same shape, with soft values in [0, 1]
        """
        y_pred = y_pred.clamp(min=eps, max=1 - eps)
        
        # Compute pt for soft labels
        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        
        # Compute focal weight
        weight = alpha * (1 - pt) ** gamma
        
        # Compute binary cross entropy
        bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        # Combine
        loss = weight * bce
        
        return loss.mean() if reduction == 'mean' else loss

    def logprob_margin_loss(log_prob, y_true, margin=1.0):
        """
        log_prob: log(p), tensor of shape (batch,)
        y_true: binary ground truth tensor (0 or 1), shape (batch,)
        margin: log-probability margin (positive float)
        """
        log_p1 = log_prob
        log_p0 = torch.log1p(-log_p1.exp() + 1e-8)  # log(1 - p) safely

        # compute margins
        margin_0 = F.relu(-log_p0 + log_p1 + margin)  # when y_true = 0
        margin_1 = F.relu(-log_p1 + log_p0 + margin)  # when y_true = 1

        return (1 - y_true) * margin_0 + y_true * margin_1
    
    def soft_margin_loss(self,y_pred, y_true, margin=0.3):
        # y_true ∈ [0,1], y_pred ∈ [0,1]
        return F.relu(1. - (2 * y_true - 1) * (y_pred - 0.5) - margin).mean()

    def focal_loss(self, y_pred, y_true, alpha=0.99, gamma=5.0, eps=1e-8):
        """
        Focal loss for binary classification.
        y_pred: predicted probabilities (after sigmoid), shape (batch_size,)
        y_true: ground truth labels (0 or 1), shape (batch_size,)
        """

        # Convert to tensors if necessary
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.tensor(y_pred, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(y_true, np.ndarray):
            y_true = torch.tensor(y_true, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        y_pred = y_pred.clamp(min=eps, max=1 - eps)

        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        focal_weight = (1 - pt) ** gamma
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        loss = -alpha_weight * focal_weight * torch.log(pt)
        return loss.mean()

    def focal_loss_pos_only_from_logprob(self,log_prob, target, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification using only the positive class and log-probabilities.
        This matches the formulation in the slide.

        Args:
            log_prob: log(p), where p is the predicted probability for class 1
            target: binary tensor with values 0 or 1
            gamma: focusing parameter
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss value
        """
        # Only apply loss where target == 1
        prob = torch.exp(log_prob)  # Convert log(p) to p
        modulating_factor = (1 - prob) ** gamma
        loss = -target * modulating_factor * log_prob  # Only targets==1 contribute

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def predict(self, dataset_predict):
            """
            Run the model in prediction mode over the given dataset.
            Returns:
                - mu_all: np.ndarray of predicted means
                - sigma_all: np.ndarray of predicted standard deviations
                - target_y_all: np.ndarray of ground truth targets
                - features_all: np.ndarray of unnormalized input features
            """
            self.model.eval()

            mu_all = []
            sigma_all = []
            target_y_all = []
            features_all = []

            dataloader = dataset_predict.dataloader

            with torch.no_grad():
                for b, batch in tqdm(
                    enumerate(dataloader),
                    total=math.ceil(len(dataloader) / dataset_predict._batch_size),
                    desc="Predicting"
                ):
                    batch_formatted = dataset_predict.format_batch_for_cnp(
                        batch, True
                    )

                    _, mu, sigma = self.model(
                        batch_formatted.query,
                        batch_formatted.target_y,
                        self.is_binary
                    )

                    # Denormalize features
                    x_norm = batch_formatted.query[1].cpu()
                    x_mean = dataset_predict.feature_mean.cpu()
                    x_std = dataset_predict.feature_std.cpu()
                    x_unnorm = x_norm * (x_std+1.e-6) + x_mean  # element-wise

                    mu_all.append(mu[0].cpu().numpy().reshape(-1, mu[0].shape[-1]))
                    sigma_all.append(sigma[0].cpu().numpy().reshape(-1, sigma[0].shape[-1]))
                    target_y_all.append(batch_formatted.target_y[0].cpu().numpy().reshape(-1, batch_formatted.target_y[0].shape[-1]))
                    features_all.append(x_unnorm.reshape(-1, x_unnorm.shape[-1])) 

            mu_all = np.concatenate(mu_all, axis=0)  
            sigma_all = np.concatenate(sigma_all, axis=0)
            target_y_all = np.concatenate(target_y_all, axis=0)
            features_all = np.concatenate(features_all, axis=0)

            return mu_all, sigma_all, target_y_all, features_all


    def aggretate_predictions(self, dataset_predict):
        mu, sigma, y, x = self.predict(dataset_predict)

        theta_size = len(self.dataset.config_file["simulation_settings"]["theta_headers"])
        theta = x[:, 0:theta_size]
        sort_idx = np.lexsort([theta[:, i] for i in reversed(range(theta_size))])

        theta_sorted = theta[sort_idx]
        mu_sorted = mu[sort_idx]
        sigma_sorted = sigma[sort_idx]
        y_sorted = y[sort_idx]

        theta_unique, inverse_indices = np.unique(theta_sorted, axis=0, return_inverse=True)
        counts = np.bincount(inverse_indices)

        mu_avg_per_theta = np.zeros((len(theta_unique), mu_sorted.shape[1]))
        np.add.at(mu_avg_per_theta, inverse_indices, mu_sorted)
        mu_avg_per_theta /= counts[:, None]

        sigma_avg_per_theta = np.zeros((len(theta_unique), sigma_sorted.shape[1]))
        np.add.at(sigma_avg_per_theta, inverse_indices, sigma_sorted)
        sigma_avg_per_theta /= counts[:, None]

        y_avg_per_theta = np.zeros((len(theta_unique), y_sorted.shape[1]))
        np.add.at(y_avg_per_theta, inverse_indices, y_sorted)
        y_avg_per_theta /= counts[:, None]

        return theta_unique, counts, mu_avg_per_theta, sigma_avg_per_theta, y_avg_per_theta

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