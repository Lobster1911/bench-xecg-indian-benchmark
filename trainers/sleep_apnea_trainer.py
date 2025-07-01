from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torch
from trainers.common_trainer import CommonTrainerDownstream
from torchmetrics import Metric
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from collections import defaultdict
import numpy as np

class TrainingSleepApnea(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.train_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)
        self.valid_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)
        self.test_feature_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)

        self.train_feature_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
        self.valid_feature_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
        self.test_feature_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)

        self.train_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)
        self.valid_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)
        self.test_acc = torchmetrics.classification.accuracy.BinaryAccuracy(ignore_index=-1)

        if not config.use_transformers:
            self.train_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
            self.valid_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
            self.test_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)

            self.train_feature_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
            self.valid_feature_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
            self.test_feature_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)

            self.train_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
            self.valid_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)
            self.test_auc = torchmetrics.AUROC(task='binary', ignore_index=-1)

            self.val_per_patient_metric = PerPatientMetric()
            self.test_per_patient_metric = PerPatientMetric()
        else: 
            # Initialize metrics for validation and test
            self.val_metric = ECG10secSegmentMetric()
            self.test_metric = ECG10secSegmentMetric()

        self.use_transformers = config.use_transformers


    def training_step(self, batch, _):
        loss, logits, preds, targets, seg_ids = self.predict_batch(batch)

        train_feature_acc = self.train_feature_acc.to(preds.device)
        train_feature_acc(preds, targets)
        self.log('train_feature_acc', train_feature_acc, prog_bar=False)
        self.log('train_loss', loss.detach().item(), prog_bar=True)

        train_feature_f1 = self.train_feature_f1.to(preds.device)
        train_feature_f1(preds, targets)
        self.log('train_feature_f1', train_feature_f1, prog_bar=False)

        if not self.use_transformers:
            preds_segment, targets_segment, logits_segment = format_to_segment(preds, targets, self.patch_size)
            train_acc = self.train_acc.to(preds.device)
            train_acc(preds_segment, targets_segment)
            self.log('train_acc', train_acc, prog_bar=True)

            train_f1 = self.train_f1.to(preds.device)
            train_f1(preds_segment, targets_segment)
            self.log('train_f1', train_f1, prog_bar=True)

            train_feature_auc = self.train_feature_auc.to(preds.device)
            train_feature_auc(logits, targets)
            self.log('train_feature_auc', train_feature_auc, prog_bar=False)

            train_auc = self.train_auc.to(preds.device)
            train_auc(logits_segment, targets_segment)
            self.log('train_auc', train_auc, prog_bar=False)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets, seg_ids = self.predict_batch(batch)

        valid_feature_acc = self.valid_feature_acc.to(preds.device)
        valid_feature_acc(preds, targets)
        self.log('val_feature_acc', valid_feature_acc, prog_bar=False)
        self.log('val_loss', loss.detach().item(), prog_bar=True)

        valid_feature_f1 = self.valid_feature_f1.to(preds.device)
        valid_feature_f1(preds, targets)
        self.log('val_feature_f1', valid_feature_f1, prog_bar=False)

        if not self.use_transformers:
            preds_segment, targets_segment, logits_segment = format_to_segment(preds, targets, self.patch_size)
            valid_acc = self.valid_acc.to(preds.device)
            valid_acc(preds_segment, targets_segment)
            self.log('val_acc', valid_acc, prog_bar=True)

            valid_f1 = self.valid_f1.to(preds.device)
            valid_f1(preds_segment, targets_segment)
            self.log('val_f1', valid_f1, prog_bar=True)

            valid_feature_auc = self.valid_feature_auc.to(preds.device)
            valid_feature_auc(logits, targets)
            self.log('val_feature_auc', valid_feature_auc, prog_bar=False)

            valid_auc = self.valid_auc.to(preds.device)
            valid_auc(logits_segment, targets_segment)
            self.log('val_auc', valid_auc, prog_bar=False)

            self.val_per_patient_metric.update(preds_segment, targets_segment, seg_ids)
        else:
            # Update the segment metric with current batch
            self.val_metric.update(preds, targets, seg_ids)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets, seg_ids = self.predict_batch(batch)

        test_feature_acc = self.test_feature_acc.to(preds.device)
        test_feature_acc(preds, targets)
        self.log('test_feature_acc', test_feature_acc, prog_bar=False)
        self.log('test_loss', loss.detach().item(), prog_bar=False)

        test_feature_f1 = self.test_feature_f1.to(preds.device)
        test_feature_f1(preds, targets)
        self.log('test_feature_f1', test_feature_f1, prog_bar=False)

        if not self.use_transformers:
            preds_segment, targets_segment, logits_segment = format_to_segment(preds, targets, self.patch_size)
            test_acc = self.test_acc.to(preds.device)
            test_acc(preds_segment, targets_segment)   
            self.log('test_acc', test_acc, prog_bar=False)
            
            test_f1 = self.test_f1.to(preds.device)
            test_f1(preds_segment, targets_segment)
            self.log('test_f1', test_f1, prog_bar=False)

            test_feature_auc = self.test_feature_auc.to(preds.device)
            test_feature_auc(logits, targets)
            self.log('test_feature_auc', test_feature_auc, prog_bar=False)

            test_auc = self.test_auc.to(preds.device)
            test_auc(logits_segment, targets_segment)
            self.log('test_auc', test_auc, prog_bar=False)

            self.test_per_patient_metric.update(preds_segment, targets_segment, seg_ids)
        else:
            # Update the segment metric with current batch
            self.test_metric.update(preds, targets, seg_ids)
 
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        if self.use_transformers:
            metrics = self.val_metric.compute()
            self.log_dict({
                'val_auc': metrics['auc_macro'],
                'val_f1': metrics['f1_macro'],
                'val_acc': metrics['accuracy_micro'],
                'val_patient_acc': metrics['patient_acc'],
                'val_patient_f1': metrics['patient_f1']
            })
        else:
            metrics = self.val_per_patient_metric.compute()
            self.log_dict({
                'val_patient_acc': metrics['patient_acc'],
                'val_patient_f1': metrics['patient_f1']
            })
        super().on_validation_epoch_end()
        
    def on_test_epoch_end(self):
        # Compute and log final test metrics
        if self.use_transformers:
            metrics = self.test_metric.compute()
            self.log_dict({
                'test_auc': metrics['auc_macro'],
                'test_f1': metrics['f1_macro'],
                'test_acc': metrics['accuracy_micro'],
                'test_patient_acc': metrics['patient_acc'],
                'test_patient_f1': metrics['patient_f1']
            })
        else:
            metrics = self.test_per_patient_metric.compute()
            self.log_dict({
                'test_patient_acc': metrics['patient_acc'],
                'test_patient_f1': metrics['patient_f1']
            })
        super().on_test_epoch_end()
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels']
        segment_ids = batch['segment_ids']  # patient ids are not used in the training, but we keep them for consistency
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).float()


        if self.use_transformers:
            loss_cls = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        else:
            targets = targets.repeat_interleave(logits.shape[-1] // targets.shape[-1], dim=-1)  # repeat targets for binary classification

            min_length = min(logits.shape[-1], targets.shape[-1])
            logits = logits[:, :min_length] 
            targets = targets[:, :min_length] 
            preds = preds[:, :min_length] 

            mask = (targets != -1)
            loss_cls = nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction='mean')

        return loss_cls, logits, preds, targets.long(), segment_ids


def format_to_segment(preds, target, patch_size, segment_size=6000):
    # preds will be [bs, seq_len, num_classes]
    # target will be [bs, seq_len, num_classes]

    patches_in_segment = segment_size // patch_size

    num_patches = preds.shape[0] if preds.ndim == 1 else preds.shape[1]

    num_segments = num_patches // patches_in_segment

    if num_patches % patches_in_segment != 0:
        # we can skip the very last part
        preds = preds[:, :-(num_patches % patches_in_segment)]
        target = target[:, :-(num_patches % patches_in_segment)]

    # take the prediction and group for patches_in_segment
    preds = preds.view(-1, num_segments, patches_in_segment)
    target = target.view(-1, num_segments, patches_in_segment)

    # reducing to segment shape
    preds_mean = (torch.mean(preds, dim=2) > 0.5).float()
    logits_mean = torch.sigmoid(torch.mean(preds, dim=2))
    target_mean = torch.max(target, dim=2)[0]

    if preds.shape != target.shape:
        raise ValueError("preds and target must have the same shape")

    return preds_mean, target_mean, logits_mean



class PerPatientMetric(Metric):
    """
    Custom metric to compute per-patient metrics based on segment-level predictions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.segment_ids = []  # Use a list to store segment IDs as strings
        # self.add_state("segment_ids", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor, segment_ids):
        """
        Update the metric with new predictions.
        
        Args:
            preds: Model predictions (logits or probabilities) - shape: (batch_size, 2) or (batch_size,)
            target: Ground truth labels - shape: (batch_size,)
            segment_ids: Segment IDs for each 10s sample - can be list of strings, torch.Tensor, or numpy array
        """
        preds = torch.sigmoid(preds)
        segment_ids = [ patient for patient, _ in segment_ids ] 
        
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
            for i in range(len(self.predictions[batch])):
                seg_id = self.segment_ids[batch][i]
                segment_preds[seg_id].extend(self.predictions[batch][i])
                segment_targets[seg_id] = self.targets[batch][i]  # Should be same for all 10s parts of same segment

        # Average predictions for each segment and collect final predictions/targets
        final_preds = []
        final_targets = []
        
        for seg_id in segment_preds.keys():
            avg_pred = np.mean(segment_preds[seg_id])
            final_preds.append(avg_pred)
            final_targets.append(segment_targets[seg_id][0])
        
        final_preds = np.array(final_preds)
        final_targets = np.array(final_targets)
        
        try:
            # Extract patient IDs and group segments by patient
            patient_segments = defaultdict(list)
            patient_predictions = defaultdict(list)
            
            for seg_id, pred in zip(segment_preds.keys(), final_preds):
                # Extract patient ID (everything before the last '_')
                patient_id = seg_id.rsplit('_', 1)[0]
                patient_segments[patient_id].append(seg_id)
                patient_predictions[patient_id].append(pred)
            
            # Calculate AHI for each patient (mean of segment predictions)
            patient_osa_true = {}
            patient_osa_pred = {}
            
            for patient_id in patient_segments.keys():
                ahi = final_targets.sum() / (len(final_targets) / 60)
                ahi_pred = final_preds.sum() / (len(final_preds) / 60)
            
                # OSA classification: AHI >= 5 (using segment predictions as AHI proxy)
                patient_osa_pred[patient_id] = 1 if ahi_pred >= 5 else 0  # 0.5 threshold for probability

                # OSA if mean of patient's segments >= 0.5 (majority of segments are positive)
                patient_osa_true[patient_id] = 1 if ahi >= 5 else 0
            
            # Convert to arrays for metric calculation
            patient_true_labels = np.array(list(patient_osa_true.values()))
            patient_pred_labels = np.array(list(patient_osa_pred.values()))
            
            # Calculate per-patient metrics
            patient_accuracy_micro = accuracy_score(patient_true_labels, patient_pred_labels)
            patient_f1_macro = f1_score(patient_true_labels, patient_pred_labels, average='macro')
            
        except Exception as e:
            print(f"Warning: Could not compute per-patient metrics - {e}")
            patient_accuracy_micro = patient_f1_macro = 0.0
        
        return {
            # Per-patient OSA metrics
            'patient_acc': patient_accuracy_micro,
            'patient_f1': patient_f1_macro
        }
    

class ECG10secSegmentMetric(Metric):
    """
    Custom metric for ECG segment classification that:
    1. Accumulates predictions for 10s segments belonging to 1min annotated segments
    2. Averages predictions by segment ID (6 predictions per segment)
    3. Computes AUC (macro), F1 (macro), and Accuracy (micro) on segment level
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
        self.predictions.append(preds.detach())
        self.targets.append(target.detach())
        self.segment_ids.extend(segment_ids)  # extend instead of append for list

    def compute(self):
        """
        Compute the final metrics after all batches have been processed.
        
        Returns:
            dict: Dictionary containing 'auc_macro', 'f1_macro', 'accuracy_micro'
        """
        # Concatenate all accumulated data
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        all_segment_ids = self.segment_ids  # Already a flat list of strings
        
        # Convert to numpy for easier processing
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Group predictions by segment ID and average them
        segment_preds = defaultdict(list)
        segment_targets = {}
        
        for i in range(len(preds_np)):
            seg_id = all_segment_ids[i]
            segment_preds[seg_id].append(preds_np[i])
            segment_targets[seg_id] = targets_np[i]  # Should be same for all 10s parts of same segment
        
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

        try:
            # Extract patient IDs and group segments by patient
            patient_segments = defaultdict(list)
            patient_predictions = defaultdict(list)
            
            for seg_id, pred in zip(segment_preds.keys(), final_preds):
                # Extract patient ID (everything before the last '_')
                patient_id = seg_id.rsplit('_', 1)[0]
                patient_segments[patient_id].append(seg_id)
                patient_predictions[patient_id].append(pred)
            
            # Calculate AHI for each patient (mean of segment predictions)
            patient_ahi = {}
            patient_osa_true = {}
            patient_osa_pred = {}
            
            for patient_id in patient_segments.keys():
                # Calculate mean AHI (average of all segment predictions for this patient)
                mean_ahi = np.mean(patient_predictions[patient_id])
                patient_ahi[patient_id] = mean_ahi
                
                # OSA classification: AHI >= 5 (using segment predictions as AHI proxy)
                patient_osa_pred[patient_id] = 1 if mean_ahi >= 0.5 else 0  # 0.5 threshold for probability
                
                # True OSA status: if any segment of this patient is positive, patient has OSA
                # (assuming segments are annotated as OSA events)
                patient_segments_targets = [segment_targets[seg_id] for seg_id in patient_segments[patient_id]]
                # OSA if mean of patient's segments >= 0.5 (majority of segments are positive)
                patient_osa_true[patient_id] = 1 if np.mean(patient_segments_targets) >= 0.5 else 0
            
            # Convert to arrays for metric calculation
            patient_true_labels = np.array(list(patient_osa_true.values()))
            patient_pred_labels = np.array(list(patient_osa_pred.values()))
            
            # Calculate per-patient metrics
            patient_accuracy_micro = accuracy_score(patient_true_labels, patient_pred_labels)
            patient_f1_macro = f1_score(patient_true_labels, patient_pred_labels, average='macro')
            
        except Exception as e:
            print(f"Warning: Could not compute per-patient metrics - {e}")
            patient_accuracy_micro = patient_f1_macro = 0.0
        
        return {
            # Segment-level metrics
            'auc_macro': auc_macro,
            'f1_macro': f1_macro,
            'accuracy_micro': accuracy_micro,
            # Per-patient OSA metrics
            'patient_acc': patient_accuracy_micro,
            'patient_f1': patient_f1_macro
        }
    
    
    def reset(self):
        """Reset the metric state."""
        self.predictions.clear()
        self.targets.clear()
        self.segment_ids.clear()

