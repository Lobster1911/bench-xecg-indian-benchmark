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
from utils.loss_utils import focal_loss


class TrainingMIMIC_LAB(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)

        self.classification_task = config.classification_task
        self.num_classes = len(config.label_list) * 3
        self.label_list = config.label_list
        self.label_keys = list(self.label_list.keys())

        self.train_accs = [
            torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        self.train_aurocs = [
            torchmetrics.AUROC(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        self.train_f1s = [
            torchmetrics.F1Score(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]

        self.val_accs = [
            torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        self.val_aurocs = [
            torchmetrics.AUROC(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        
        self.val_f1s = [
            torchmetrics.F1Score(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]

        self.test_accs = [
            torchmetrics.Accuracy(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        self.test_aurocs = [
            torchmetrics.AUROC(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]
        self.test_f1s = [
            torchmetrics.F1Score(num_classes=num_classes, average='macro', task='multiclass', top_k=1, ignore_index=-1).to(self.device)
            for _, num_classes in self.label_list.items()
        ]

    def update_metrics(self, accs, aurocs, f1s, targets, logits):
        for i, (k, n_class) in enumerate(self.label_list.items()):
            accs[i].to(logits.device)
            aurocs[i].to(logits.device)
            f1s[i].to(logits.device)

            if n_class == 3:
                preds = logits[:, i].argmax(dim=-1)
                accs[i].update(preds, targets[:, i])
                aurocs[i].update(logits[:, i], targets[:, i])
                f1s[i].update(preds, targets[:, i])
            elif n_class == 2:
                preds = logits[:, i, 1:].argmax(dim=-1)
                # do not consider the class below
                ignore = targets[:, i] == -1
                reduced_target = targets[:, i] -1
                reduced_target[ignore] = -1

                accs[i].update(preds, reduced_target)
                aurocs[i].update(logits[:, i, 1:], reduced_target)
                f1s[i].update(preds, reduced_target)
            else:
                raise ValueError(f"Number of classes {n_class} not supported")

    def log_metrics(self, accs, aurocs, f1s, step='train'):
        avg_acc = torch.mean(torch.tensor([acc.compute().mean() for acc in accs]))
        avg_auroc = torch.mean(torch.tensor([auroc.compute().mean() for auroc in aurocs]))
        avg_f1 = torch.mean(torch.tensor([f1.compute().mean() for f1 in f1s]))

        self.log(f"{step}_acc_avg", avg_acc, prog_bar=True)
        self.log(f"{step}_auroc_avg", avg_auroc, prog_bar=True)
        self.log(f"{step}_f1_avg", avg_f1, prog_bar=False)

    def training_step(self, batch, _):
        loss, logits, targets = self.predict_batch(batch)

        self.update_metrics(self.train_accs, self.train_aurocs, self.train_f1s, targets, logits)

        self.log_metrics(self.train_accs, self.train_aurocs, self.train_f1s, step='train')

        self.log("train_loss", loss, prog_bar=True)

        return loss
    
    def validation_step(self, batch, _):
        loss, logits, targets = self.predict_batch(batch)

        self.update_metrics(self.val_accs, self.val_aurocs, self.val_f1s, targets, logits)

        self.log_metrics(self.val_accs, self.val_aurocs, self.val_f1s, step='val')

        self.log("val_loss", loss, prog_bar=True)

        return loss
            
    def test_step(self, batch, _):
        loss, logits, targets = self.predict_batch(batch)

        self.update_metrics(self.test_accs, self.test_aurocs, self.test_f1s, targets, logits)

        # log the accuracy for each label
        accs = [acc.compute() for acc in self.test_accs]
        for i, acc in enumerate(accs):
            self.log(f"{self.label_keys[i]}/test_acc", acc, prog_bar=True)
        # get the average accuracy
        avg_acc = torch.mean(torch.tensor(accs))
        self.log("test_acc_avg", avg_acc, prog_bar=True)

        # log the auroc for each label
        aurocs = [auroc.compute() for auroc in self.test_aurocs]
        for i, auroc in enumerate(aurocs):
            self.log(f"{self.label_keys[i]}/test_auroc", auroc, prog_bar=True)
        # get the average auroc
        avg_auroc = torch.mean(torch.tensor(aurocs))
        self.log("test_auroc_avg", avg_auroc, prog_bar=True)

        # log the f1 score for each label
        f1s = [f1.compute() for f1 in self.test_f1s]
        for i, f1 in enumerate(f1s):
            self.log(f"{self.label_keys[i]}/test_f1", f1, prog_bar=True)
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

        def argmax_with_ignore(targets, ignore_value=-1, dim=-1):
            all_ignore_mask = torch.all(targets == ignore_value, dim=dim)
            argmax_result = torch.argmax(targets, dim=dim)
            return torch.where(all_ignore_mask,  torch.full_like(argmax_result, -1), argmax_result)

        targets_indices = argmax_with_ignore(targets, dim=-1)  # [bs, num_tasks]

        logits_flat = logits.view(-1, 3)
        targets_flat = targets_indices.view(-1)  # [bs * num_tasks]

        # print(f"Logits shape: {logits.shape}, Targets shape: {targets.shape}, Logits flat shape: {logits_flat.shape}, Targets flat shape: {targets_flat.shape}")

        loss = nn.functional.cross_entropy(logits_flat, targets_flat, weight=self.weights, ignore_index=-1)
    
        return loss, logits, targets_indices
    
