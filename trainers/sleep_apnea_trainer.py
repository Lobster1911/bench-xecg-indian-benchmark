from torch import nn
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torch
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from collections import defaultdict
import numpy as np

from trainers.common_trainer import CommonTrainerDownstream


class TrainingSleepApnea(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.train_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1, average='macro')
        self.valid_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1, average='macro')
        self.test_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1, average='macro')

        self.train_acc = torchmetrics.Accuracy(task='binary', ignore_index=-1)
        self.valid_acc = torchmetrics.Accuracy(task='binary', ignore_index=-1)
        self.test_acc = torchmetrics.Accuracy(task='binary', ignore_index=-1)

        self.train_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
        self.valid_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
        self.test_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)

        self.multiple_predictions_per_segment = config.window_size < 60  # if window size is more than 1 minute, we have multiple predictions per segment

        if self.multiple_predictions_per_segment:
            self.train_segment_metric = ECGSegmentMetric()
            self.val_segment_metric = ECGSegmentMetric()
            self.test_segment_metric = ECGSegmentMetric()

    def training_step(self, batch, _):
        loss, logits, preds, targets, segment_ids = self.predict_batch(batch)
        self.log('train_loss', loss.detach().item(), prog_bar=True)

        if self.multiple_predictions_per_segment:
            self.train_segment_metric.update(preds, targets, segment_ids)
        else:
            train_acc = self.train_acc.to(preds.device)
            train_acc(preds, targets)
            self.log('train_acc', train_acc, prog_bar=False)

            train_f1 = self.train_f1.to(preds.device)
            train_f1(preds, targets)
            self.log('train_f1', train_f1, prog_bar=False)

            train_auc = self.train_auc.to(preds.device)
            train_auc(logits, targets)
            self.log('train_auc', train_auc, prog_bar=False)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets, segment_ids = self.predict_batch(batch)
        self.log('val_loss', loss.detach().item(), prog_bar=False)

        if self.multiple_predictions_per_segment:
            self.val_segment_metric.update(preds, targets, segment_ids)
        else: 
            val_acc = self.valid_acc.to(preds.device)
            val_acc(preds, targets)
            self.log('val_acc', val_acc, prog_bar=False)

            val_f1 = self.valid_f1.to(preds.device)
            val_f1(preds, targets)
            self.log('val_f1', val_f1, prog_bar=True)

            val_auc = self.valid_auc.to(preds.device)
            val_auc(logits, targets)
            self.log('val_auc', val_auc, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets, segment_ids = self.predict_batch(batch)
        self.log('test_loss', loss.detach().item(), prog_bar=False)


        if self.multiple_predictions_per_segment:
            self.test_segment_metric.update(preds, targets, segment_ids)
        else:  
            test_acc = self.test_acc.to(preds.device)
            test_acc(preds, targets)
            self.log('test_acc', test_acc, prog_bar=False)

            test_f1 = self.test_f1.to(preds.device)
            test_f1(preds, targets)
            self.log('test_f1', test_f1, prog_bar=False)

            test_auc = self.test_auc.to(preds.device)
            test_auc(logits, targets)
            self.log('test_auc', test_auc, prog_bar=False)

        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        if self.multiple_predictions_per_segment:
            metrics = self.val_segment_metric.compute()
            self.log_dict({
                'val_auc': metrics['auc_macro'],
                'val_f1': metrics['f1_macro'],
                'val_acc': metrics['accuracy_micro'],
            })
        super().on_validation_epoch_end()
        
    def on_test_epoch_end(self):
        # Compute and log final test metrics
        if self.multiple_predictions_per_segment:
            metrics = self.test_segment_metric.compute()
            self.log_dict({
                'test_auc': metrics['auc_macro'],
                'test_f1': metrics['f1_macro'],
                'test_acc': metrics['accuracy_micro'],
            })
        super().on_test_epoch_end()
    
            
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels']
        segment_ids = batch['segment_ids']  # patient ids are not used in the training, but we keep them for consistency

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        # mask = (targets != -1)
        logits = self.model(x)
        if logits.ndim > 2:
            logits = logits.squeeze(-1)

        preds = (torch.sigmoid(logits) > 0.5).float()

        loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        return loss_cls, logits, preds, targets.long(), segment_ids


class ECGSegmentMetric(Metric):
    """
    Custom metric for ECG segment classification that:
    1. Accumulates predictions for 10s segments belonging to 1min annotated segments
    2. Averages predictions by segment ID (6 predictions per segment)
    3. Computes AUC (macro), F1 (macro), and Accuracy (micro) on segment level
    """
    
    def __init__(self, ignore_index=None, **kwargs):
        super().__init__(**kwargs)
        
        self.ignore_index = ignore_index
        # State variables to accumulate data across batches
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        # self.add_state("segment_ids", default=[], dist_reduce_fx=None)  # None for string lists
        self.segment_ids = []  # Use a list to store segment IDs as strings
        
    def update(self, preds: torch.Tensor, target: torch.Tensor, segment_ids):
        """
        Update the metric with new predictions.
        
        Args:
            preds: Model predictions (logits or probabilities) - shape: (batch_size, 2) or (batch_size,)
            target: Ground truth labels - shape: (batch_size,)
            segment_ids: Segment IDs for each 10s sample - can be list of strings, torch.Tensor, or numpy array
        """
        preds = torch.sigmoid(preds.squeeze())
        segment_ids = [ patient + '_' + str(seg_id) for patient, seg_id in segment_ids ] 

        # Store predictions, targets, and segment IDs
        self.predictions.append(preds.detach().cpu())
        self.targets.append(target.detach().cpu())
        self.segment_ids.append(segment_ids)


    def compute(self):
        """
        Compute the final metrics after all batches have been processed.
        
        Returns:
            dict: Dictionary containing 'auc_macro', 'f1_macro', 'accuracy_micro'
        """
        # Group predictions by segment ID and average them
        segment_preds = defaultdict(list)
        segment_targets = {}
        
        for batch in range(len(self.predictions)):
            if self.predictions[batch].ndim == 0:
                continue
            for i in range(len(self.predictions[batch])):
                # check target is not -1
                if  self.ignore_index is not None and self.targets[batch][i].item() == self.ignore_index:
                    continue
                seg_id = self.segment_ids[batch][i]
                segment_preds[seg_id].append(self.predictions[batch][i].item())
                segment_targets[seg_id] = self.targets[batch][i].item()  # Should be same for all 10s parts of same segment

        # Average predictions for each segment and collect final predictions/targets
        final_preds = []
        final_targets = []
        
        for seg_id in segment_preds.keys():
            avg_pred = np.mean(segment_preds[seg_id])
            final_preds.append(avg_pred)
            final_targets.append(segment_targets[seg_id])
        
        final_preds = np.array(final_preds)
        final_targets = np.array(final_targets)
        
        # Convert probabilities to binary predictions for F1 and accuracy
        binary_preds = (final_preds > 0.5).astype(int)
        
        # Calculate metrics
        try:
            # AUC (macro) - for binary classification, macro and micro AUC are the same
            auc_macro = roc_auc_score(final_targets, final_preds)
            # F1 (macro)
            f1_macro = f1_score(final_targets, binary_preds, average='macro')
            # Accuracy (micro) - for binary classification, this is just regular accuracy
            accuracy_micro = accuracy_score(final_targets, binary_preds)

        except ValueError as e:
            # Handle edge cases (e.g., all predictions are same class)
            print(f"Warning: Could not compute some metrics - {e}")
            auc_macro = f1_macro = accuracy_micro = 0.0
        
        return {
            # Segment-level metrics
            'auc_macro': auc_macro,
            'f1_macro': f1_macro,
            'accuracy_micro': accuracy_micro,
            # Per-patient OSA metrics
        }
    
    
    def reset(self):
        """Reset the metric state."""
        self.predictions.clear()
        self.targets.clear()
        self.segment_ids.clear()

