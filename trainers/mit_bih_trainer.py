from torch import optim, nn
import torchmetrics
import torchmetrics.classification
import torchmetrics.classification.accuracy
import torchmetrics.classification.precision_recall
import torchmetrics.classification.specificity
import numpy as np
import torch
import trainers.common as common
from trainers.common_trainer import CommonTrainerDownstream
import os
import matplotlib.pyplot as plt
import lightning.pytorch as pl

class TrainingMIT_BIH(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)
            
        self.plot_test_predictions = config.plot_test_predictions

        self.predict_no_hb = config.predict_no_hb
        if self.predict_no_hb:
            self.num_classes -= 1

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.test_acc_no_avg = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average=None, ignore_index=-1, top_k=1)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=-1, top_k=1)
        self.valid_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=-1, top_k=1)
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average=None, ignore_index=-1, top_k=1)

        self.train_auroc = torchmetrics.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1, task='multiclass')
        self.valid_auroc = torchmetrics.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1, task='multiclass')
        self.test_auroc = torchmetrics.AUROC(num_classes=self.num_classes, compute_on_step=False, ignore_index=-1, task='multiclass')

        self.single_hb = config.single_hb

        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.Specificity(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_spec = torchmetrics.Specificity(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.val_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.val_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)

    def training_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.train_acc = self.train_acc.to(preds.device)
        self.train_acc(preds.flatten(), targets.flatten())

        self.train_f1 = self.train_f1.to(preds.device)
        self.train_f1(preds.flatten(), targets.flatten())

        self.log('train_loss', loss_cls.detach().item(), prog_bar=True)


        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)

        # auroc
        self.train_auroc = self.train_auroc.to(logits.device)
        self.train_auroc(logits, targets)
        self.log("train_auroc", self.train_auroc)

        return loss_cls 
    
    def validation_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.valid_acc = self.valid_acc.to(preds.device)
        self.valid_acc(preds.flatten(), targets.flatten())

        self.valid_f1 = self.valid_f1.to(preds.device)
        self.valid_f1(preds.flatten(), targets.flatten())

        self.log('val_loss', loss_cls.detach().item(), prog_bar=True)

        self.log('val_acc', self.valid_acc, prog_bar=True)
        self.log('val_f1', self.valid_f1, prog_bar=True)

        # specificity
        self.val_spec = self.val_spec.to(preds.device)
        self.val_spec(preds, targets)
        self.log('val_specificity/N', self.val_spec[0], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/S', self.val_spec[1], prog_bar=False, metric_attribute='val_spec')
        self.log('val_specificity/V', self.val_spec[2], prog_bar=False, metric_attribute='val_spec')
        if self.num_classes == 5:
            self.log('val_specificity/F', self.val_spec[3], prog_bar=False, metric_attribute='val_spec')
            self.log('val_specificity/Q', self.val_spec[4], prog_bar=False, metric_attribute='val_spec')

        # sensitivity
        self.val_recall = self.val_recall.to(preds.device)
        self.val_recall(preds, targets)
        self.log('val_sensitivity/N', self.val_recall[0], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/S', self.val_recall[1], prog_bar=False, metric_attribute='val_recall')
        self.log('val_sensitivity/V', self.val_recall[2], prog_bar=False, metric_attribute='val_recall')
        if self.num_classes == 5:
            self.log('val_sensitivity/F', self.val_recall[3], prog_bar=False, metric_attribute='val_recall')
            self.log('val_sensitivity/Q', self.val_recall[4], prog_bar=False, metric_attribute='val_recall')

        # ppv
        self.val_precision = self.val_precision.to(preds.device)
        self.val_precision(preds, targets)
        self.log('val_ppv/N', self.val_precision[0], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/S', self.val_precision[1], prog_bar=False, metric_attribute='val_precision')
        self.log('val_ppv/V', self.val_precision[2], prog_bar=False, metric_attribute='val_precision')
        if self.num_classes == 5:
            self.log('val_ppv/F', self.val_precision[3], prog_bar=False, metric_attribute='val_precision')
            self.log('val_ppv/Q', self.val_precision[4], prog_bar=False, metric_attribute='val_precision')

        # auroc
        self.valid_auroc = self.valid_auroc.to(logits.device)
        self.valid_auroc(logits, targets)
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        return loss_cls
            
    def test_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)

        self.test_acc = self.test_acc.to(preds.device)
        self.test_acc(preds.flatten(), targets.flatten())

        self.test_acc_no_avg = self.test_acc_no_avg.to(preds.device)
        self.test_acc_no_avg(preds.flatten(), targets.flatten())

        self.test_f1 = self.test_f1.to(preds.device)
        self.test_f1(preds.flatten(), targets.flatten())

        self.log("test_loss", loss_cls.detach().item())


        self.log("test_acc", self.test_acc)

        # accuracy
        self.log("test_acc/N", self.test_acc_no_avg[0], metric_attribute='test_acc')
        self.log("test_acc/S", self.test_acc_no_avg[1], metric_attribute='test_acc')
        self.log("test_acc/V", self.test_acc_no_avg[2], metric_attribute='test_acc')
        if self.num_classes == 5:
            self.log("test_acc/F", self.test_acc_no_avg[3], metric_attribute='test_acc')
            self.log("test_acc/Q", self.test_acc_no_avg[4], metric_attribute='test_acc')

        # f1
        self.log("test_f1/N", self.test_f1[0], metric_attribute='test_f1')
        self.log("test_f1/S", self.test_f1[1], metric_attribute='test_f1')
        self.log("test_f1/V", self.test_f1[2], metric_attribute='test_f1')
        if self.num_classes == 5:
            self.log("test_f1/F", self.test_f1[3], metric_attribute='test_f1')
            self.log("test_f1/Q", self.test_f1[4], metric_attribute='test_f1')
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2] + self.test_f1[3] + self.test_f1[4]) / 5, metric_attribute='test_f1')
        else:
            self.log("test_f1/mean", (self.test_f1[0] + self.test_f1[1] + self.test_f1[2]) / 3, metric_attribute='test_f1')

        # specificity
        self.test_spec = self.test_spec.to(preds.device)
        self.test_spec(preds, targets)
        self.log("test_specificity/N", self.test_spec[0], metric_attribute='test_spec')
        self.log("test_specificity/S", self.test_spec[1], metric_attribute='test_spec')
        self.log("test_specificity/V", self.test_spec[2], metric_attribute='test_spec')
        if self.num_classes == 5:
            self.log("test_specificity/F", self.test_spec[3], metric_attribute='test_spec')
            self.log("test_specificity/Q", self.test_spec[4], metric_attribute='test_spec')
            self.log("test_specificity/mean", (self.test_spec[0] + self.test_spec[1] + self.test_spec[2] + self.test_spec[3] + self.test_spec[4]) / 5, metric_attribute='test_spec')
        else:
            self.log("test_specificity/mean", (self.test_spec[0] + self.test_spec[1] + self.test_spec[2]) / 3, metric_attribute='test_spec')

        # sensitivity
        self.test_recall = self.test_recall.to(preds.device)
        self.test_recall(preds, targets)
        self.log("test_sensitivity/N", self.test_recall[0], metric_attribute='test_recall')
        self.log("test_sensitivity/S", self.test_recall[1], metric_attribute='test_recall')
        self.log("test_sensitivity/V", self.test_recall[2], metric_attribute='test_recall')
        if self.num_classes == 5:
            self.log("test_sensitivity/F", self.test_recall[3], metric_attribute='test_recall')
            self.log("test_sensitivity/Q", self.test_recall[4], metric_attribute='test_recall')
            self.log("test_sensitivity/mean", (self.test_recall[0] + self.test_recall[1] + self.test_recall[2] + self.test_recall[3] + self.test_recall[4]) / 5, metric_attribute='test_recall')
        else:
            self.log("test_sensitivity/mean", (self.test_recall[0] + self.test_recall[1] + self.test_recall[2]) / 3, metric_attribute='test_recall')

        # ppv
        self.test_precision = self.test_precision.to(preds.device)
        self.test_precision(preds, targets)
        self.log("test_ppv/N", self.test_precision[0], metric_attribute='test_precision')
        self.log("test_ppv/S", self.test_precision[1], metric_attribute='test_precision')
        self.log("test_ppv/V", self.test_precision[2], metric_attribute='test_precision')
        if self.num_classes == 5:
            self.log("test_ppv/F", self.test_precision[3], metric_attribute='test_precision')
            self.log("test_ppv/Q", self.test_precision[4], metric_attribute='test_precision')
            self.log("test_ppv/mean", (self.test_precision[0] + self.test_precision[1] + self.test_precision[2] + self.test_precision[3] + self.test_precision[4]) / 5, metric_attribute='test_precision')
        else:
            self.log("test_ppv/mean", (self.test_precision[0] + self.test_precision[1] + self.test_precision[2]) / 3, metric_attribute='test_precision')

        # auroc  
        self.test_auroc = self.test_auroc.to(logits.device)
        self.test_auroc(logits, targets)
        self.log("test_auroc", self.test_auroc)

        return loss_cls 
    
    def on_validation_epoch_end(self):
        super().on_test_epoch_end()

        if self.plot_test_predictions:
            try:
                idx_1 = np.random.randint(0, len(self.trainer.val_dataloaders.dataset))
                idx_2 = np.random.randint(0, len(self.trainer.val_dataloaders.dataset))
                sample_1 = self.trainer.val_dataloaders.dataset[idx_1]
                sample_2 = self.trainer.val_dataloaders.dataset[idx_2]
                log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir is not None else 'figs/'
                img_1 = self.plot_mit_bih_pred(sample_1, log_dir, 'mit_1_val')
                img_2 = self.plot_mit_bih_pred(sample_2, log_dir, 'mit_2_val')

                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.log_image(key="reconstructions_val", images=[img_1, img_2])
            except Exception as e:
                # print stack trace
                import traceback
                traceback.print_exc()
                print(f"Error plotting R-peaks: {e}")

    
    def on_train_epoch_end(self):
        super().on_test_epoch_end()

        if self.plot_test_predictions:
            try:
                idx_1 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))
                idx_2 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))
                sample_1 = self.trainer.train_dataloader.dataset[idx_1]
                sample_2 = self.trainer.train_dataloader.dataset[idx_2]
                log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir is not None else 'figs/'
                img_1 = self.plot_mit_bih_pred(sample_1, log_dir, 'mit_1_train')
                img_2 = self.plot_mit_bih_pred(sample_2, log_dir, 'mit_2_train')

                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.log_image(key="reconstructions_train", images=[img_1, img_2])
            except Exception as e:
                # print stack trace
                import traceback
                traceback.print_exc()
                print(f"Error plotting R-peaks: {e}")


    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels'].long()

        if self.linear_probing:
            self.model.set_eval_linear_probing()

        if self.predict_no_hb:
            targets = targets + 1

        if self.single_hb:
            cls = self.model(x) # [bs, 1, num_classes]
            targets = targets.squeeze()
            loss_cls = nn.functional.cross_entropy(cls, targets, weight=self.weights, ignore_index=-1)
            preds = torch.argmax(cls, dim=-1)
        else:
            cls = self.model(x).permute(0, 2, 1)
            #print('cls shape:', cls.shape)
            #print('target shape:', targets.shape)
            loss_cls = nn.functional.cross_entropy(cls, targets, weight=self.weights, ignore_index=-1)
            preds = torch.argmax(cls, dim=-2)

        if self.use_focal_loss:
            pt = torch.exp(-loss_cls)
            alpha = 2.
            gamma = .25
            loss_cls = (alpha * (1-pt)**gamma * loss_cls)

        if self.predict_no_hb:
            preds = preds - 1
            preds[preds == -1] = 0
            targets = targets - 1
            cls = cls[..., 1:, :]
        
        return loss_cls, preds, targets, cls

    def plot_mit_bih_pred(self, sample, logdir, name):
        with torch.no_grad():
            signal = torch.from_numpy(sample['signal']).to(self.device).unsqueeze(0)
            targets = torch.from_numpy(sample['label']).to(self.device).unsqueeze(0)

            predicted = self.model(signal)
            targets = targets.unfold(1, self.model.patch_size, self.model.patch_size).max(dim=-1)[0].long()

            # consider max 2000 time samples for plotting
            # if signal.shape[1] > max_length:
            #     signal = signal[:, :max_length, :]
            #    target = target[:, :max_length]

            # targets = target.unfold(1, model.patch_size, model.patch_size).max(dim=-1)[0].long()
            fig, ax = plt.subplots(figsize=(25, 5))

            to_plot = signal[:, :, 1].cpu().squeeze().numpy() if signal.ndim > 2 else signal.cpu().squeeze().numpy()

            ax.plot(to_plot, label='Original Signal')

            # Plot vertical lines at each patch
            for j in range(0, signal.shape[1], self.model.patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # inside each patch plot the prediction above and the target below

            for i in range(targets.shape[1]):
                patch_start = i * self.model.patch_size
                patch_end = patch_start + self.model.patch_size

                # get the max index of the prediction
                pred_class = torch.argmax(predicted[0, i]).item() - 1
                ax.text((patch_start + patch_end) / 2, 0.5,
                        f'{get_label(pred_class)}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=7,
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                # plot the target class below the patch
                target_class = targets[0, i].item()
                ax.text((patch_start + patch_end) / 2, -0.5,
                        f'{get_label(target_class)}',  
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=7,
                        color='orange',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                
                # color the patch background if the prediction is correct or not
                if target_class != -1:
                    ax.axvspan(patch_start, patch_end, color='green' if pred_class == target_class else 'red', alpha=0.1)
                
            ax.set_title('MIT-BIH ECG Signal with Predictions and Targets')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.legend()

            # mkdir if it does not exist
            os.makedirs(f'{logdir}/epoch_{self.current_epoch}', exist_ok=True)

            path = f'{logdir}/epoch_{self.current_epoch}/r_peaks_{name}.png'
            plt.savefig(path)
            plt.close()
            return path
    

def get_label(label):
    if label == 0: return 'N'
    if label == 1: return 'S'
    if label == 2: return 'V'
    if label == 3: return 'F'
    if label == 4: return 'Q'
    if label == -1: return 'X'
    else: raise ValueError(f'Unknown label {label}')