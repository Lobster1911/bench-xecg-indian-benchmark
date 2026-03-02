import os
import torch
import numpy as np
from torch import nn
from typing import Optional

import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.precision_recall
from torchmetrics import Metric
import lightning.pytorch as pl
import matplotlib.pyplot as plt

from trainers.common_trainer import CommonTrainerDownstream



class TrainingRPeak(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.sampling_freq = config.sampling_freq
        self.original_freq = config.original_freq
        
        self.train_rec = torchmetrics.classification.precision_recall.BinaryRecall()
        self.valid_rec = torchmetrics.classification.precision_recall.BinaryRecall()
        self.test_rec = torchmetrics.classification.precision_recall.BinaryRecall()

        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.valid_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.valid_acc = torchmetrics.classification.BinaryAccuracy()
        self.test_acc = torchmetrics.classification.BinaryAccuracy()

        self.train_auprc = torchmetrics.classification.BinaryAveragePrecision()
        self.valid_auprc = torchmetrics.classification.BinaryAveragePrecision()
        self.test_auprc = torchmetrics.classification.BinaryAveragePrecision()

        self.val_distance_150 = RPeakDistanceMetric(
            orig_freq=self.original_freq,
            pred_freq=self.sampling_freq,
            threshold_window=150
        )

        self.val_distance_20 = RPeakDistanceMetric(
            orig_freq=self.original_freq,
            pred_freq=self.sampling_freq,
            threshold_window=20
        )

        self.test_distance_150 = RPeakDistanceMetric(
            orig_freq=self.original_freq,
            pred_freq=self.sampling_freq,
            threshold_window=150
        )

        self.test_distance_20 = RPeakDistanceMetric(
            orig_freq=self.original_freq,
            pred_freq=self.sampling_freq,
            threshold_window=20
        )

        self.plot_predictions = config.plot_predictions

    def training_step(self, batch, _):
        loss_r_peak_pos, r_peak_pos, r_peaks, r_peaks_orig = self.predict_batch(batch)

        self.train_rec = self.train_rec.to(r_peak_pos.device)
        self.train_rec(r_peak_pos, r_peaks)

        self.train_f1 = self.train_f1.to(r_peak_pos.device)
        self.train_f1(r_peak_pos, r_peaks)

        self.train_acc = self.train_acc.to(r_peak_pos.device)
        self.train_acc(r_peak_pos, r_peaks)

        self.train_auprc = self.train_auprc.to(r_peak_pos.device)
        self.train_auprc(r_peak_pos, r_peaks)


        self.log('train_loss', loss_r_peak_pos.detach().item(), prog_bar=True)
        self.log('train_rec', self.train_rec, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_auprc', self.train_auprc, prog_bar=True)

        return loss_r_peak_pos

    def validation_step(self, batch, _):
        loss_r_peak_pos, r_peak_pos, r_peaks, r_peaks_orig = self.predict_batch(batch)

        self.valid_rec(r_peak_pos, r_peaks)
        self.valid_f1(r_peak_pos, r_peaks)
        self.valid_acc(r_peak_pos, r_peaks)
        self.valid_auprc(r_peak_pos, r_peaks)
        self.val_distance_150.update(r_peak_pos, r_peaks_orig)
        self.val_distance_20.update(r_peak_pos, r_peaks_orig)

        self.log('val_loss', loss_r_peak_pos.detach().item(), prog_bar=True)
        self.log('val_rec', self.valid_rec, prog_bar=True)
        self.log('val_f1', self.valid_f1, prog_bar=True)
        self.log('val_acc', self.valid_acc, prog_bar=True)
        self.log('val_auprc', self.valid_auprc, prog_bar=True)

        return loss_r_peak_pos
    
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        # Log the average distance metric
        avg_distance_150 = self.val_distance_150.compute()
        self.log('val_avg_distance', avg_distance_150['avg_distance'], prog_bar=False)
        self.log('val_avg_distance_rp', avg_distance_150['avg_distance_rp'], prog_bar=False)
        self.log('val_avg_total_distance', avg_distance_150['avg_total_distance'], prog_bar=True)
        self.log('val_ppv_150', avg_distance_150['ppv'], prog_bar=False)
        self.log('val_tpr_150', avg_distance_150['tpr'], prog_bar=False)
        self.log('val_f1_150', avg_distance_150['f1'], prog_bar=True)
        self.val_distance_150.reset()

        avg_distance_20 = self.val_distance_20.compute()

        self.log('val_ppv_20', avg_distance_20['ppv'], prog_bar=False)
        self.log('val_tpr_20', avg_distance_20['tpr'], prog_bar=False)
        self.log('val_f1_20', avg_distance_20['f1'], prog_bar=True)
        self.val_distance_20.reset()

        self.plot_samples_if_needed(self.trainer.val_dataloaders, step='val')

    def test_step(self, batch, _):
        loss_r_peak_pos, r_peak_pos, r_peaks, r_peaks_orig = self.predict_batch(batch)

        self.test_rec = self.test_rec.to(r_peak_pos.device)
        self.test_rec(r_peak_pos, r_peaks)

        self.test_f1 = self.test_f1.to(r_peak_pos.device)
        self.test_f1(r_peak_pos, r_peaks)

        self.test_acc = self.test_acc.to(r_peak_pos.device)
        self.test_acc(r_peak_pos, r_peaks)

        self.test_auprc = self.test_auprc.to(r_peak_pos.device)
        self.test_auprc(r_peak_pos, r_peaks)

        self.test_distance_150 = self.test_distance_150.to(r_peak_pos.device)
        self.test_distance_150.update(r_peak_pos, r_peaks_orig)

        self.test_distance_20 = self.test_distance_20.to(r_peak_pos.device)
        self.test_distance_20.update(r_peak_pos, r_peaks_orig)

        self.log("test_loss", loss_r_peak_pos.detach().item())
        self.log("test_rec", self.test_rec)
        self.log("test_f1", self.test_f1)
        self.log("test_acc", self.test_acc)
        self.log("test_auprc", self.test_auprc)

        return loss_r_peak_pos

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

        # Log the average distance metric
        avg_distance_150 = self.test_distance_150.compute()
        self.log('test_avg_distance', avg_distance_150['avg_distance'])
        self.log('test_avg_distance_rp', avg_distance_150['avg_distance_rp'])
        self.log('test_avg_total_distance', avg_distance_150['avg_total_distance'])
        self.log('test_ppv_150', avg_distance_150['ppv'])
        self.log('test_tpr_150', avg_distance_150['tpr'])
        self.log('test_f1_150', avg_distance_150['f1'])

        self.test_distance_150.reset()

        avg_distance_20 = self.test_distance_20.compute()
        self.log('test_ppv_20', avg_distance_20['ppv'])
        self.log('test_tpr_20', avg_distance_20['tpr'])
        self.log('test_f1_20', avg_distance_20['f1'])   
        self.test_distance_20.reset()

        self.plot_samples_if_needed(self.trainer.test_dataloaders, step='test')


    def plot_samples_if_needed(self, dataloader, step='train'):
        if self.plot_predictions:
            try:
                # sample_1 = self.trainer.test_dataloaders.dataset[0]
                sample_1 = dataloader.dataset[0]
                sample_2 = dataloader.dataset[1]
                log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir is not None else 'figs/'
                img_1 = plot_r_peaks(sample_1, self.model, self.sampling_freq, self.device, log_dir, self.current_epoch, 'r_peaks_1', step=step)
                img_2 = plot_r_peaks(sample_2, self.model, self.sampling_freq, self.device, log_dir, self.current_epoch, 'r_peaks_2', step=step)

                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.log_image(key=f"reconstructions_{step}", images=[img_1, img_2])
            except Exception as e:
                # print stack trace
                import traceback
                traceback.print_exc()
                print(f"Error plotting R-peaks: {e}")

    def predict_batch(self, batch):
        x = batch["signals"]
        # print(f"Signal shape: {x.shape}")
        r_peaks = batch['r_peak'] # [bs, seq_len]
        # print(f"R-peaks shape: {r_peaks.shape}")
        r_peaks_orig = batch['r_peak_orig']
        # print(f"R-peaks original: {r_peaks_orig}")

        if self.linear_probing:
            self.model.set_eval_linear_probing()

        r_peak_pos= self.model(x)
        # print(f"R-peaks prediction shape: {r_peak_pos.shape}")
        r_peak_pos = r_peak_pos.view(r_peak_pos.shape[0], -1)

        # print(r_peak_pos.shape, r_peaks.shape)

        min_len = min(r_peak_pos.shape[1], r_peaks.shape[1])
        r_peak_pos = r_peak_pos[:, :min_len]
        r_peaks = r_peaks[:, :min_len]

        # print(f"Prediction: {r_peak_pos.shape}, Target: {r_peaks.shape}, weights: {self.weights}")

        pos_weight = torch.tensor(self.patch_size, dtype=torch.float32).to(r_peak_pos.device) if self.weights is not None else None
        loss_r_peak_pos = nn.functional.binary_cross_entropy_with_logits(r_peak_pos, r_peaks, pos_weight=pos_weight)

        r_peak_pos = torch.sigmoid(r_peak_pos)  # Apply sigmoid to get probabilities
        r_peak_pos = (r_peak_pos > 0.5).float()

        return loss_r_peak_pos, r_peak_pos, r_peaks, r_peaks_orig


# given all the heartbeats in the batch, find the closest predicted r-peak to each heartbeat and calculate a time distance

class RPeakDistanceMetric(Metric):
    """
    Custom TorchMetric to calculate the average distance between predicted R-peaks 
    and actual R-peaks in time domain.
    
    This metric handles frequency conversion and calculates the temporal distance
    between predictions and ground truth R-peaks.
    """
    
    def __init__(
        self,
        orig_freq: float,
        pred_freq: float,
        dist_sync_on_step: bool = False,
        process_group: Optional = None,
        dist_sync_fn = None,
        threshold_window: Optional[float] = None
    ):
        """
        Args:
            orig_freq: Original sampling frequency of the signal (Hz)
            pred_freq: Prediction sampling frequency (Hz) 
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        self.orig_freq = orig_freq
        self.pred_freq = pred_freq
        # threshold in milliseconds -> convert to seconds and divide by two to have a window around the r-peak
        self.threshold_window = (threshold_window / 1000) / 2

        # Metric state
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_distance_rp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_predictions", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("positives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, r_peaks_orig: list) -> None:
        """
        Update metric state with new predictions and original R-peaks.
        
        Args:
            preds: Predicted R-peaks [batch_size, sequence_length] with 1s at R-peaks
            r_peaks_orig: Original R-peak indices for this batch [batch_size, max_peaks] 
                         (use -1 or 0 for padding where no peak exists)
            batch_start_indices: Starting index in original frequency for each batch sample [batch_size]
        """
        batch_size = preds.shape[0]
        
        total_distance = 0.0
        total_distance_rp = 0.0
        total_predictions = 0

        true_positives = 0
        false_positives = 0
        positives = 0
        
        for i in range(batch_size):
            # Extract peak indices from predictions
            # I can create a list where I store the time indices of the r-peaks in seconds instead of index
            # print(f"R-peaks original: {r_peaks_orig[i]}")
            list_r_peaks = torch.tensor([int(r) / self.orig_freq for r in r_peaks_orig[i] if not torch.isnan(r)]).to(preds.device)
            # print(f"List R-peaks secodns: {list_r_peaks}")
            # get the indices of non-zero preds
            # print(f"Preds: {preds[i]}")
            pred_peaks = torch.nonzero(preds[i], as_tuple=False).squeeze(-1).to(preds.device)
            # print(f"Predicted R-peaks indices: {pred_peaks}")
            list_pred_peaks = pred_peaks.float() / self.pred_freq  # Convert to seconds 
            # print(f"Predicted R-peaks seconds: {list_pred_peaks}")
            # print(f"Predicted R-peaks seconds: {list_pred_peaks}")

            # calculate the distance for all the r-peaks in the batch
            if len(list_pred_peaks) == 0:
                continue

            if list_pred_peaks.numel() == 0 or list_r_peaks.numel() == 0:
                # Se uno dei due è vuoto, assegna tensori vuoti per evitare crash
                min_distances = torch.tensor([]).to(list_pred_peaks)
                min_dist_r_peaks = torch.ones_like(list_r_peaks) * preds[i].shape[0] 
                matched_rpeaks = torch.tensor([]).to(list_pred_peaks)
            else:
                distances = torch.abs(list_pred_peaks.unsqueeze(1) - list_r_peaks.unsqueeze(0))
                # print(f"Distances: {distances.shape}")
                # Find the minimum distance for each predicted peak

                # for every prediction I have a measure of how far is from the nearest r-peak
                min_distances, _ = torch.min(distances, dim=1) 
                # print(f"Min distances: {min_distances.shape}")
                min_dist_r_peaks, _ = torch.min(distances, dim=0)  # For each original R-peak, find the closest prediction
                # print(f"Min distances: {min_distances.shape}")
                # Get predicted-to-R-peak assignments
                matched_mask = min_distances <= self.threshold_window  # shape [num_preds]
                matched_rpeaks = distances[matched_mask].argmin(dim=1)  # Get the indices of the closest R-peaks for each prediction


            total_distance += min_distances.sum().item()
            total_distance_rp += min_dist_r_peaks.sum().item()
            total_predictions += len(list_pred_peaks)


            # Unique matches → equivalent to set-based count
            true_positives += len(torch.unique(matched_rpeaks))
            if list_pred_peaks.numel() != 0:
                false_positives += (~matched_mask).sum().item()
            positives += len(list_r_peaks)

        self.total_distance += total_distance
        self.total_distance_rp += total_distance_rp
        self.num_predictions += total_predictions
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.positives += positives
    
    def compute(self) -> dict:
        """
        Compute the final metric values.
        
        Returns:
            Dictionary containing:
            - avg_distance: Average distance between predicted and actual R-peaks (seconds)
            - match_rate: Percentage of predictions that found valid matches
            - total_predictions: Total number of predictions
            - total_matches: Total number of valid matches
        """
        
        avg_distance = (self.total_distance / self.num_predictions) * 1000 # convert to milliseconds
        avg_distance_rp = (self.total_distance_rp / self.num_predictions) * 1000 # convert to milliseconds

        avg_tot_dist = (avg_distance + avg_distance_rp) / 2

        tpr = self.true_positives / self.positives if self.positives > 0 else 0
        ppv = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0

        f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else 0

        return {
            "avg_distance": avg_distance,
            "avg_distance_rp": avg_distance_rp,
            "avg_total_distance": avg_tot_dist,
            "total_predictions": self.num_predictions,
            "total_matches": self.true_positives,
            "ppv": ppv,
            "tpr": tpr,
            "f1": f1
        }
    
    def reset(self):
        self.total_distance = torch.tensor(0.0)
        self.total_distance_rp = torch.tensor(0.0)
        self.num_predictions = torch.tensor(0)
        self.true_positives = torch.tensor(0)
        self.false_positives = torch.tensor(0)
        self.positives = torch.tensor(0)
        self.false_negatives = torch.tensor(0)
        self.true_negatives = torch.tensor(0)
        return super().reset()



def plot_r_peaks(sample, model, sampling_freq, device, logdir, epoch, name, step='train'):
    with torch.no_grad():
        signal = sample['signals'].to(device).unsqueeze(0)
        r_peaks = sample['r_peak'].to(device).unsqueeze(0)
        # print(f"Signal shape: {signal.shape}")

        # Get the original R-peaks
        # print(f"R-peaks shape: {r_peaks.shape}")

        # Get the predicted R-peaks
        r_peak_pos = model(signal)
        r_peak_pos = r_peak_pos.view(r_peak_pos.shape[0], -1)
        r_peak_pos = torch.sigmoid(r_peak_pos) > 0.5

        # consider max 2000 time samples for plotting
        if signal.shape[1] > 10 * sampling_freq:
            signal = signal[:, :10 * sampling_freq, :]
            r_peak_pos = r_peak_pos[:, :10 * sampling_freq]
            r_peaks = r_peaks[:, :10 * sampling_freq]

        # print(f"Predicted R-peaks shape: {r_peak_pos.shape}")

        # Plot the original and predicted R-peaks
        fig, ax = plt.subplots(figsize=(25, 5))
        to_plot = signal[:, :, 1].cpu().squeeze().numpy() if signal.ndim > 2 else signal.cpu().squeeze().numpy()

        ax.plot(to_plot)
        # Plot vertical lines for predicted R-peaks
        pred_peaks = np.where(r_peak_pos.cpu().squeeze().numpy())[0]
        for peak in pred_peaks:
            ax.axvline(peak, color='darkorange', linestyle='solid', linewidth=1.5, label='Predicted R-peak' if peak == pred_peaks[0] else "", alpha=0.5)
        
        gts = np.where(r_peaks.cpu().squeeze().numpy())[0]
        for gt in gts:
            ax.axvline(gt, color='darkgreen', linestyle='--', linewidth=1.5, label='Ground Truth R-peak' if gt == gts[0] else "", alpha=0.8)

        # ax.set_title(f'R-peaks Prediction - {name}')
        ax.set_xlabel('Timepoints', fontdict={'size': 24})
        ax.set_ylabel('Amplitude', fontdict={'size': 24})

        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.legend(fontsize=24)
        plt.tight_layout()

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}/{step}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/{step}/r_peaks_{name}.png'
        plt.savefig(path, dpi=300)
        plt.close()
        return path