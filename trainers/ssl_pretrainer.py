import lightning as L
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss, embedding_cross_entropy_loss, masked_cosine_loss
from torch.nn import functional as F
from utils.plot_utils import plot_reconstruction, plot_generation
import numpy as np
import torch
import lightning
import trainers.common as common
import torch.distributed
from loss import KoLeoLoss, MCRLoss
from threadpoolctl import threadpool_limits
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
            len_train_dataset,
            config,
            knn_train_dataloader,
            knn_val_dataloader
        ):
        super().__init__()
        self.lr = config.lr
        self.reconstruction_lr = config.reconstruction_lr
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.wd = config.wd
        self.use_scheduler = config.use_scheduler
        self.patch_size = config.patch_size
        self.epochs = config.epochs
        self.loss_type = config.loss_type
        self.mask_ratio = config.mask_ratio
        # self.config = config
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.num_epochs_warm_restart = config.num_epochs_warm_restart
        self.sched_decay_factor = config.sched_decay_factor
        self.grad_loss_lambda = config.grad_loss_lambda
        self.min_max_loss_lambda = config.min_max_loss_lambda
        self.pretraining_strategy = config.strategy
        self.start_train_head_at_epoch = config.start_train_head_at_epoch

        self.ema_0 = config.ema_0
        self.ema_1 = config.ema_1

        self.use_sim_dino = config.use_sim_dino
        self.use_koleo_regularization = config.use_koleo_regularization and not self.use_sim_dino

        self.centering = config.centering
        self.teacher_temp = config.teacher_temp
        self.stud_temp = config.stud_temp

        if self.use_sim_dino: self.mcr_loss = MCRLoss(eps=0.05)
        if self.use_koleo_regularization: self.koleo_reg = KoLeoLoss()

        if self.model.use_teacher_student:
            self.model.init_teacher()
            self.automatic_optimization=False

        self.knn_train_dataloader = knn_train_dataloader
        self.knn_val_dataloader = knn_val_dataloader

    def training_step(self, batch, _):
        losses = self.reconstruct_batch(batch, step='train')
        rec_loss, jepa_loss = losses['reconstruction_loss'], losses['teacher_student_loss']

        if self.model.use_teacher_student:
            opt_core, opt_head = self.optimizers()
            sched_core, sched_head = self.lr_schedulers()

            train_head = self.current_epoch >= self.start_train_head_at_epoch
            
            opt_core.zero_grad()
            self.manual_backward(jepa_loss, retain_graph=train_head)
            self.clip_gradients(opt_core, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            opt_core.step()
            sched_core.step()

            if train_head:
                opt_head.zero_grad()
                self.manual_backward(rec_loss)
                opt_head.step()
                sched_head.step()

            self.update_teacher()
            return
        else:
            return rec_loss
    
    @torch.no_grad()
    def update_teacher(self):
        steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
        num_training_steps = steps_per_epoch * self.epochs
        beta = self.ema_0 + self.global_step * (self.ema_1 - self.ema_0) / num_training_steps
        self.log('teacher_beta', beta, prog_bar=False)
        self.update_module(self.model._teacher, self.model, beta)
        
    def update_module(self, teacher_module, student_module, beta):
        for param_s, param_t in zip(student_module.parameters(), teacher_module.parameters()):
            param_t.data = param_t.data * beta + (1.0 - beta) * param_s.data

    def update_param(self, teacher_param, student_param, beta):
        teacher_param.data = teacher_param.data * beta + (1.0 - beta) * student_param.data

    def validation_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='val')
        return loss
    
    def test_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='test')
        return loss
    
    def on_train_epoch_end(self):
        """
        When the training loop ends, some representative plots from different classes are saved on wandb
        """
        if self.logger is None:
            return super().on_validation_epoch_end()
        
        sample_1 = self.trainer.train_dataloader.dataset[0]
        sample_2 = self.trainer.train_dataloader.dataset[-42]

        # get two random samples from the training dataset
        idx_3 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))
        idx_4 = np.random.randint(0, len(self.trainer.train_dataloader.dataset))

        sample_3 = self.trainer.train_dataloader.dataset[idx_3]
        sample_4 = self.trainer.train_dataloader.dataset[idx_4]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir


        img_1 = plot_reconstruction(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_2 = plot_reconstruction(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_3 = plot_reconstruction(sample_3, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_3_random', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_4 = plot_reconstruction(sample_4, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_4_random', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="reconstructions_train", images=[img_1, img_2, img_3, img_4])

        if self.pretraining_strategy == 'next_token_prediction':
            img_1 = plot_generation(sample_1, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_1')
            img_2 = plot_generation(sample_2, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_2')
            img_3 = plot_generation(sample_3, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_3_random')
            img_4 = plot_generation(sample_4, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_4_random')
            if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
                self.logger.log_image(key="generations_train", images=[img_1, img_2, img_3, img_4])

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        """
        When the validation loop ends, some representative plots from different classes are saved on wandb
        """
        self.knn_evaluation()

        if self.logger is None:
            return super().on_validation_epoch_end()
        
        # save the plots of the reconstruction for some samples
        sample_s = self.trainer.val_dataloaders.dataset[115]
        sample_v = self.trainer.val_dataloaders.dataset[91]
        sample_t = self.trainer.val_dataloaders.dataset[23]
        sample_n = self.trainer.val_dataloaders.dataset[0]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir

        img_s = plot_reconstruction(sample_s, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_s', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_v = plot_reconstruction(sample_v, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_v', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_t = plot_reconstruction(sample_t, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_t', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        img_n = plot_reconstruction(sample_n, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_n', training_strategy=self.pretraining_strategy, mask_ratio=self.mask_ratio)
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key="reconstructions", images=[img_s, img_v, img_t, img_n])

        if self.pretraining_strategy == 'next_token_prediction':
            img_s = plot_generation(sample_s, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_s')
            img_v = plot_generation(sample_v, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_v')
            img_t = plot_generation(sample_t, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_t')
            img_n = plot_generation(sample_n, self.model, self.patch_size, self.device, log_dir, self.current_epoch, 'sample_n')
            if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
                self.logger.log_image(key="generations", images=[img_s, img_v, img_t, img_n])
        return super().on_validation_epoch_end()
    
    def reconstruct_batch(self, batch, step):
        global_signals = batch["global_signals"]
        local_signals = batch["local_signals"]
        batch_size, seq_len, num_leads = global_signals[0].shape

        global_out = [self.model(x, masking=True) for x in global_signals]
        global_out_teacher = [self.model.teacher_fwd(x) for x in global_signals]
        local_out = [self.model(x, masking=False, reconstruct=False) for x in local_signals]

        # compute the loss and use the gradients only when it is needed
        teacher_student_loss = None
       
        # patch based loss
        masks = [out['mask'] for out in global_out]

        padding_masks = [(sig != 0.).flip(1).cumsum(dim=1).flip(1) == 0 for sig in global_signals]
        padding_masks_patched = [m.view(batch_size, m.shape[1] // self.patch_size, self.patch_size, num_leads) for m in padding_masks]
        padding_masks_patched = [m.max(dim=-1)[0] for m in padding_masks_patched]
        combined_padding_mask = torch.cat(padding_masks_patched, dim=1).max(dim=-1)[0].flatten(0, 1)

        patched_masks = [m.view(batch_size, m.shape[1] // self.patch_size, self.patch_size) for m in masks]
        combined_mask = torch.cat(patched_masks, dim=1).max(dim=-1)[0].flatten(0, 1)
        # i do not want to predict where masking is applied to masked tokens
        combined_mask = combined_mask * combined_padding_mask

        if self.use_sim_dino:
            cls_tok_stud_g = torch.stack([g['cls'] for g in global_out] + [l['cls'] for l in local_out], dim=0)
            cls_tok_teacher_g = torch.stack([g['cls'] for g in global_out_teacher], dim=0)
        
            compression_term, expansion_term = self.mcr_loss(cls_tok_stud_g, cls_tok_teacher_g)
            self.log(f"{step}_compression_term", compression_term.item(), prog_bar=False)
            self.log(f"{step}_expansion_term", expansion_term.item(), prog_bar=False)

            stud_embeddings = torch.cat([out['patches'] for out in global_out], dim=1).flatten(0, 1)
            teacher_embeddings = torch.cat([out_t['patches'] for out_t in global_out_teacher], dim=1).flatten(0, 1)

            # simplified dino uses only the mse between the embeddigns
            patch_loss = masked_cosine_loss(stud_embeddings, teacher_embeddings, reduction='mean', mask=combined_mask)
            self.log(f"{step}_patch_loss", patch_loss.item(), prog_bar=True)

            teacher_student_loss = compression_term + expansion_term + patch_loss

        else:
            stud_embeddings = torch.cat([out['patches'] for out in global_out], dim=1).flatten(0, 1)
            teacher_embeddings = torch.cat([out_t['patches'] for out_t in global_out_teacher], dim=1).flatten(0, 1)
            # i use first cls tokens of the second augmented signal to match the cls of the first augmented signal
            patch_loss = embedding_cross_entropy_loss(stud_embeddings, teacher_embeddings, reduction='mean', mask=combined_mask, centering=self.centering, stud_temp=self.stud_temp, teacher_temp=self.teacher_temp)

            cross_cls_loss = []
            for i, t_out in enumerate(global_out_teacher):
                for j, s_g_out in enumerate(global_out):
                    if i == j: continue
                    cross_cls_loss.append(embedding_cross_entropy_loss(s_g_out['cls'], t_out['cls'], reduction='mean', mask=combined_mask, centering=self.centering, stud_temp=self.stud_temp, teacher_temp=self.teacher_temp))
                for s_l_out in local_out:
                    cross_cls_loss.append(embedding_cross_entropy_loss(s_l_out['cls'], t_out['cls'], reduction='mean', mask=combined_mask, centering=self.centering, stud_temp=self.stud_temp, teacher_temp=self.teacher_temp))

            cross_cls_loss = torch.stack(cross_cls_loss, dim=0).mean()
            # TODO add losses

        
        if self.use_koleo_regularization:
            koleo_loss1 = self.koleo_reg(global_out[0]['cls']) # on cls tokens
            koleo_loss2 = self.koleo_reg(global_out[1]['cls'])
            koleo_loss = (koleo_loss1 + koleo_loss2) / 2
            teacher_student_loss = teacher_student_loss + koleo_loss * 0.1
            self.log(f"{step}_koleo_loss", koleo_loss.item(), prog_bar=True)

        self.log(f"{step}_dino_loss", teacher_student_loss.item(), prog_bar=True)


        rank_me1 = self.rank_me(global_out[0]['cls'])
        rank_me2 = self.rank_me(global_out[1]['cls'])
        self.log(f"{step}_rank_me", ((rank_me1 + rank_me2) / 2).item(), prog_bar=True)

        # log norm of output
        with torch.no_grad():
            if self.use_sim_dino:
                norm = torch.norm(global_out[0]['patches'], dim=-1)
            else: 
                norm = torch.norm(global_out[0]['patches_after_head'], dim=-1)
            norm = norm.mean()
            self.log(f"{step}_norm_emb", norm.item(), prog_bar=False)

            # log the mean cosine similarity between all samples in the batch
            cos_sim = torch.nn.functional.cosine_similarity(global_out[0]['cls'].unsqueeze(1), global_out[1]['cls'].unsqueeze(0), dim=-1)
            # zero the diagonal
            cos_sim2 = torch.nn.functional.cosine_similarity(global_out[1]['cls'].unsqueeze(1), global_out[0]['cls'].unsqueeze(0), dim=-1)
            mask = torch.eye(cos_sim.shape[0], device=cos_sim.device).bool()
            cos_sim_diff = (cos_sim[~mask].mean() + cos_sim2[~mask].mean()) / 2
            cos_sim_same = (cos_sim2[mask].mean() + cos_sim[mask].mean()) / 2
            self.log(f"{step}_cos_sim_different_samples", cos_sim_diff.item(), prog_bar=False)
            self.log(f"{step}_cos_sim_same_samples", cos_sim_same.mean().item(), prog_bar=False)
        
        if self.pretraining_strategy == 'masked_token_prediction' and self.model.use_teacher_student:
            nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(global_out[0]['reconstruction'], global_signals[0], mask = None) #  out['mask'])
            nrmse2, mse2, mae2, grad2, min_max2 = self.calculate_metrics_reconstruction(global_out[1]['reconstruction'], global_signals[1], mask=None) #, out2['mask'])
            nrmse, mse, mae, grad, min_max = (nrmse + nrmse2) / 2, (mse + mse2) / 2, (mae + mae2) / 2, (grad + grad2) / 2, (min_max + min_max2) / 2
        else:
            raise ValueError(f"Pretraining strategy {self.pretraining_strategy} still to be implemented completely")
            nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(out['reconstruction'], x, out['mask'])

        loss = torch.tensor(0.0, device=self.device)

        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse
        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True)
        self.log(f"{step}_mse", mse.item(), prog_bar=False)
        self.log(f"{step}_mae", mae.item(), prog_bar=False)
        self.log(f"{step}_grad", grad.item(), prog_bar=False)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False)
        
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=False)

        return {'reconstruction_loss': loss, 'teacher_student_loss': teacher_student_loss}
    
    @torch.no_grad()
    def rank_me(self, tensor, eps=1e-8):
        if not torch.isfinite(tensor).all():
            return torch.tensor(0.0, device=tensor.device)
        try:
            _, S, _ = torch.linalg.svd(tensor, full_matrices=False)  # shape: (min(N, D),)

            # Normalize singular values to get a probability distribution
            S_norm = S / (S.sum() + eps)

            # Entropy of the distribution
            entropy = -torch.sum(S_norm * torch.log(S_norm + eps))

            # Effective rank
            rank_me = torch.exp(entropy)
            return rank_me / tensor.shape[0]
        except:
            return torch.tensor(0.0, device=self.device)
    
    def calculate_metrics_reconstruction(self, rec, target, mask):
        batch_size, tokens_num, channels = target.shape
        if mask is not None:
            patched_mask = mask.view(batch_size, tokens_num // self.patch_size, self.patch_size)
            mask = mask.view(batch_size, tokens_num, 1).repeat_interleave(channels, dim=-1)
        else:
            patched_mask = None

        nrmse = np.inf
    
        if 'min_max' in self.loss_type:
            min_max = masked_min_max_loss(rec, target, patch_size=self.patch_size, mask=patched_mask)

        if 'mae' in self.loss_type:
            mae = masked_mae_loss(rec, target, mask=mask)
        else:
            with torch.no_grad(): mae = masked_mae_loss(rec, target, mask=mask)

        if 'grad' in self.loss_type:
            grad = gradient_loss(rec, target, mask=mask)
        else:
            with torch.no_grad(): grad = gradient_loss(rec, target, mask=mask)

        if 'mse' in self.loss_type:
            mse = masked_mse_loss(rec, target, reduction='mean', mask=mask)
        else:
            with torch.no_grad(): mse = masked_mse_loss(rec, target, reduction='mean', mask=mask)
   
        # calculate the normalized root squared error only for the first token prediction
        with torch.no_grad():
            nrmse = torch.sqrt(mse) / (target.max() - target.min())

        return nrmse, mse, mae, grad, min_max

    def next_token_prediction(self, batch):
        x = self.pad(batch["signal"])

        reconstruction, out_teacher, last_emb = self.model(x)

        x = x[:, self.patch_size:].squeeze()
        reconstruction = reconstruction[:, :-self.patch_size]

        if self.model.use_teacher_student:
            out_teacher = out_teacher[:, 1:, :] # [bs, seq_len -1, num_hiddens]
            last_emb = last_emb[:, :-1, :] # [bs, seq_len -1, num_hiddens]
            return x, reconstruction, out_teacher, last_emb
        
        return x, reconstruction, None, None
    
    @torch.no_grad()
    def knn_evaluation(self):
        self.model.eval()
        all_features = []
        all_labels = []
        # loop the knn dataloader to get the embeddings
        for sample in self.knn_train_dataloader:
            out = self.model(sample["signals"].to(self.device), masking=False, reconstruct=False)
            all_features.append(out['cls'].detach().cpu())
            all_labels.append(sample['class_labels'].detach().cpu())

        X_train = torch.cat(all_features).numpy()
        y_train = torch.cat(all_labels).numpy()
        
        knn = KNeighborsClassifier(n_neighbors=5)
        model = OneVsRestClassifier(knn)

        with threadpool_limits(limits=1):
            model.fit(X_train, y_train)

            # get the validation part
            all_features_val = []
            all_labels_val = []

            for sample in self.knn_val_dataloader:
                out = self.model(sample["signals"].to(self.device))
                all_features_val.append(out['cls'].detach().cpu())
                all_labels_val.append(sample['class_labels'].detach().cpu())

            X_val = torch.cat(all_features_val).numpy()
            y_val = torch.cat(all_labels_val).numpy()

            y_pred = model.predict(X_val)
            x_pred = model.predict(X_train)

            f1 = f1_score(y_val, y_pred, average='macro')
            f1_train = f1_score(y_train, x_pred, average='macro')
            self.log('downstream_knn_ptbxl_f1', f1, prog_bar=True)
            self.log('downstream_knn_ptbxl_f1_train', f1_train, prog_bar=False)

    def get_params(self):
        return self.model.trainable_parameters()
    
    def get_lr(self):
        return self.lr
    
    def get_reconstruction_lr(self):
        return self.reconstruction_lr

    def configure_optimizers(self):
        if self.model.use_teacher_student:
            return common.configure_optimizer_teacher_student(self)
        else:
            return common.configure_optimizers(self)

