from torch import optim, nn
import lightning as L
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.f_beta
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
from trainers.common_trainer import CommonTrainerDownstream
from utils.train_utils import focal_loss


class TrainingMIMIC_LAB(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.classification_task = config.classification_task
        self.num_classes = len(config.label_list) * 3
        self.label_list = config.label_list

        self.train_accs = [
            torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]
        self.train_aurocs = [
            torchmetrics.AUROC(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))  
        ]
        self.train_f1s = [
            torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]

        self.val_accs = [
            torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]
        self.val_aurocs = [
            torchmetrics.AUROC(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]
        
        self.val_f1s = [
            torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]

        self.test_accs = [
            torchmetrics.Accuracy(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]
        self.test_aurocs = [
            torchmetrics.AUROC(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]
        self.test_f1s = [
            torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _ in range(len(config.label_list))
        ]


    def training_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        # update the accuracy metrics
        for i, acc in enumerate(self.train_accs):
            acc.to(loss.device)
            acc.update(preds[:, i], targets[:, i])
        # get the average accuracy
        avg_acc = torch.mean(torch.tensor([acc.compute() for acc in self.train_accs]))
        self.log("train_acc_avg", avg_acc, prog_bar=True)
        
        # update aurocs
        for i, auroc in enumerate(self.train_aurocs):
            auroc.to(loss.device)
            auroc.update(logits[:, i], targets[:, i])
        # get the average auroc
        avg_auroc = torch.mean(torch.tensor([auroc.compute() for auroc in self.train_aurocs]))
        self.log("train_auroc_avg", avg_auroc, prog_bar=True)

        # update f1 scores
        for i, f1 in enumerate(self.train_f1s):
            f1.to(loss.device)
            f1.update(preds[:, i], targets[:, i])
        # get the average f1 score
        avg_f1 = torch.mean(torch.tensor([f1.compute() for f1 in self.train_f1s]))
        self.log("train_f1_avg", avg_f1, prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        # update the accuracy metrics
        for i, acc in enumerate(self.val_accs):
            acc.to(loss.device)
            acc.update(preds[:, i], targets[:, i])
        # get the average accuracy
        avg_acc = torch.mean(torch.tensor([acc.compute() for acc in self.val_accs]))
        self.log("val_acc_avg", avg_acc, prog_bar=False)

        # update aurocs
        for i, auroc in enumerate(self.val_aurocs):
            auroc.to(loss.device)
            auroc.update(logits[:, i], targets[:, i])
        # get the average auroc
        avg_auroc = torch.mean(torch.tensor([auroc.compute() for auroc in self.val_aurocs]))
        self.log("val_auroc_avg", avg_auroc, prog_bar=True)

        # update f1 scores
        for i, f1 in enumerate(self.val_f1s):
            f1.to(loss.device)
            f1.update(preds[:, i], targets[:, i])
        # get the average f1 score
        avg_f1 = torch.mean(torch.tensor([f1.compute() for f1 in self.val_f1s]))
        self.log("val_f1_avg", avg_f1, prog_bar=False)

        self.log("val_loss", loss, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, preds, targets = self.predict_batch(batch)

        # update the accuracy metrics
        for i, acc in enumerate(self.test_accs):
            acc.to(loss.device)
            acc.update(preds[:, i], targets[:, i])
        # log the accuracy for each label
        accs = [acc.compute() for acc in self.test_accs]
        for i, acc in enumerate(accs):
            self.log(f"{self.label_list[i]}/test_acc", acc, prog_bar=True)
        # get the average accuracy
        avg_acc = torch.mean(torch.tensor(accs))
        self.log("test_acc_avg", avg_acc, prog_bar=True)

        # update aurocs
        for i, auroc in enumerate(self.test_aurocs):
            auroc.to(loss.device)
            auroc.update(logits[:, i], targets[:, i]) 
        # log the auroc for each label
        aurocs = [auroc.compute() for auroc in self.test_aurocs]
        for i, auroc in enumerate(aurocs):
            self.log(f"{self.label_list[i]}/test_auroc", auroc, prog_bar=True)
        # get the average auroc
        avg_auroc = torch.mean(torch.tensor(aurocs))
        self.log("test_auroc_avg", avg_auroc, prog_bar=True)

        # update f1 scores
        for i, f1 in enumerate(self.test_f1s):
            f1.to(loss.device)
            f1.update(preds[:, i], targets[:, i])
        # log the f1 score for each label
        f1s = [f1.compute() for f1 in self.test_f1s]
        for i, f1 in enumerate(f1s):
            self.log(f"{self.label_list[i]}/test_f1", f1, prog_bar=True)
        # get the average f1 score
        avg_f1 = torch.mean(torch.tensor(f1s))
        self.log("test_f1_avg", avg_f1, prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)
        return loss
            
    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels']
        # get one hot encoding

        if self.linear_probing: 
            self.model.set_eval_linear_probing()

        logits = self.model(x) # [bs, num_classes = num_labels * 3]
        logits = logits.view(logits.shape[0], self.num_classes // 3, 3)  # [bs, num_classes // 3, 3]
        targets = targets.view(targets.shape[0], self.num_classes // 3, 3)  # [bs, num_classes // 3, 3]

        targets_indices = torch.argmax(targets, dim=-1)  # [bs, num_tasks]

        logits_flat = logits.view(-1, 3)
        targets_flat = targets_indices.view(-1)  # [bs * num_tasks]

        # print(f"Logits shape: {logits.shape}, Targets shape: {targets.shape}, Logits flat shape: {logits_flat.shape}, Targets flat shape: {targets_flat.shape}")

        loss = nn.functional.cross_entropy(logits_flat, targets_flat, weight=self.weights, ignore_index=-1)
        preds = torch.argmax(logits, dim=-1)
    
        return loss, logits, preds, targets_indices
