from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torch
from trainers.common_trainer import CommonTrainerDownstream
 

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

        self.train_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
        self.valid_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)
        self.test_f1 = torchmetrics.F1Score(task='binary', ignore_index=-1)



    def training_step(self, batch, _):
        loss, _, preds, targets = self.predict_batch(batch)

        train_feature_acc = self.train_feature_acc.to(preds.device)
        train_feature_acc(preds, targets)
        self.log('train_feature_acc', train_feature_acc, prog_bar=False)
        self.log('train_loss', loss.detach().item(), prog_bar=True)

        train_feature_f1 = self.train_feature_f1.to(preds.device)
        train_feature_f1(preds, targets)
        self.log('train_feature_f1', train_feature_f1, prog_bar=False)

        preds_segment, targets_segment = format_to_segment(preds, targets, self.patch_size)
        train_acc = self.train_acc.to(preds.device)
        train_acc(preds_segment, targets_segment)
        self.log('train_acc', train_acc, prog_bar=True)

        train_f1 = self.train_f1.to(preds.device)
        train_f1(preds_segment, targets_segment)
        self.log('train_f1', train_f1, prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, _, preds, targets = self.predict_batch(batch)

        valid_feature_acc = self.valid_feature_acc.to(preds.device)
        valid_feature_acc(preds, targets)
        self.log('val_feature_acc', valid_feature_acc, prog_bar=False)
        self.log('val_loss', loss.detach().item(), prog_bar=True)

        valid_feature_f1 = self.valid_feature_f1.to(preds.device)
        valid_feature_f1(preds, targets)
        self.log('val_feature_f1', valid_feature_f1, prog_bar=False)

        preds_segment, targets_segment = format_to_segment(preds, targets, self.patch_size)
        valid_acc = self.valid_acc.to(preds.device)
        valid_acc(preds_segment, targets_segment)
        self.log('val_acc', valid_acc, prog_bar=True)

        valid_f1 = self.valid_f1.to(preds.device)
        valid_f1(preds_segment, targets_segment)
        self.log('val_f1', valid_f1, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, _, preds, targets = self.predict_batch(batch)

        test_feature_acc = self.test_feature_acc.to(preds.device)
        test_feature_acc(preds, targets)
        self.log('test_feature_acc', test_feature_acc, prog_bar=False)
        self.log('test_loss', loss.detach().item(), prog_bar=False)

        test_feature_f1 = self.test_feature_f1.to(preds.device)
        test_feature_f1(preds, targets)
        self.log('test_feature_f1', test_feature_f1, prog_bar=False)

        preds_segment, targets_segment = format_to_segment(preds, targets, self.patch_size)
        test_acc = self.test_acc.to(preds.device)
        test_acc(preds_segment, targets_segment)   
        self.log('test_acc', test_acc, prog_bar=False)
        
        test_f1 = self.test_f1.to(preds.device)
        test_f1(preds_segment, targets_segment)
        self.log('test_f1', test_f1, prog_bar=False)
 
        return loss
            
    
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x).squeeze()
        preds = (torch.sigmoid(logits) > 0.5).float().squeeze()

        mask = (targets != -1)
        loss_cls = nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask], weight=self.weights)
        
        return loss_cls, logits, preds, targets.long()


def format_to_segment(preds, target, patch_size, segment_size=6000):
    # preds will be [bs, seq_len, num_classes]
    # target will be [bs, seq_len, num_classes]

    patches_in_segment = segment_size // patch_size
    num_patches = preds.shape[1]
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
    target_mean = torch.max(target, dim=2)[0]

    if preds.shape != target.shape:
        raise ValueError("preds and target must have the same shape")
    
    return preds_mean, target_mean


