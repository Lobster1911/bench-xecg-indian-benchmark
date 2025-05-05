import lightning as L
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss, embedding_cross_entropy_loss, vicreg_loss
from torch.nn import functional as F
from utils.plot_utils import plot_reconstruction, plot_generation
import numpy as np
import torch
import lightning
import trainers.common as common
import torch.distributed
from loss import KoLeoLoss, MCRLoss

# define the LightningModule
class PretrainedxLSTMNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
            len_train_dataset,
            config
        ):
        super().__init__()
        self.lr = config.lr
        self.lr_reconstruction = config.lr_reconstruction
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

        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        losses = self.reconstruct_batch(batch, step='train')
        rec_loss, jepa_loss = losses['reconstruction_loss'], losses['teacher_student_loss']

        if self.model.use_teacher_student:
            opt_core, opt_head = self.optimizers()
            sched_core, sched_head = self.lr_schedulers()

            train_head = self.current_epoch >= self.start_train_head_at_epoch
            
            opt_core.zero_grad()
            self.manual_backward(jepa_loss, retain_graph=train_head)
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

        self.update_module(self.model._teacher, self.model, beta)
        #self.update_module(self.model.patch_embedding, self.model._patch_embedding_teacher, beta)
        #self.update_param(self.model.cls_token, self.model._cls_token_teacher, beta)
        #self.update_param(self.model.reg_token, self.model._reg_tokens_teacher, beta)

        #if not self.use_sim_dino:
        #    self.update_module(self.model.layer_norm, self.model._layer_norm_teacher, beta)
        #    self.update_module(self.model.dino_head, self.model._dino_head_teacher, beta)
        #    self.update_module(self.model.ibot_head, self.model._ibot_head_teacher, beta)
        
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
        signal = batch["signals"]
        batch_size, _, _ = signal.shape
        if self.pretraining_strategy == 'next_token_prediction':
            x, rec, out_teacher, last_emb = self.next_token_prediction(batch)
            mask = None #TODO fix this and set the mask accordingly to padded tokens
        if self.pretraining_strategy == 'masked_token_prediction':
            # masking is automatically done inside this function
            # x will be masked with the inverse of the mask
            # so the loss  function will automatically skip the masked values
            # mask is 1 for masked and 0 for non masked
            x = self.pad(signal)
            out = self.model(x, masking=True)
            if self.model.use_teacher_student:
                out_teacher = self.model.teacher_fwd(x)
                signal2 = batch["signals_2"]
                x2 = self.pad(signal2)
                out2 = self.model(x2, masking=True)
                out2_teacher = self.model.teacher_fwd(x2)
                
        # compute the loss and use the gradients only when it is needed
        teacher_student_loss = None
        if self.model.use_teacher_student:
            mask1 = out['mask']
            patched_mask1 = mask1.view(batch_size, x.shape[1] // self.patch_size, self.patch_size)
            mask2 = out2['mask']
            patched_mask2 = mask2.view(batch_size,  x2.shape[1] // self.patch_size, self.patch_size)
            combined_mask = torch.cat([patched_mask1, patched_mask2], dim=0).max(dim=-1)[0].flatten(0, 1)

            if self.use_sim_dino:
                stud_embeddings = torch.cat([out['patches'], out2['patches']], dim=0).flatten(0, 1)
                teacher_embeddings = torch.cat([out_teacher['patches'], out2_teacher['patches']], dim=0).flatten(0, 1)

                cls_tokens_stud = torch.cat([out2['cls'], out['cls']], dim=0)
                cls_tokens_teacher = torch.cat([out_teacher['cls'], out2_teacher['cls']], dim=0)

                # simplified dino uses only the mse between the embeddigns
                cross_cls_loss = masked_mse_loss(cls_tokens_stud, cls_tokens_teacher, reduction='mean', mask=None)
                patch_loss = masked_mse_loss(stud_embeddings, teacher_embeddings, reduction='mean', mask=combined_mask)
            else:
                stud_embeddings = torch.cat([out['patches_after_head'], out2['patches_after_head']], dim=0).flatten(0, 1)
                teacher_embeddings = torch.cat([out_teacher['patches_after_head'], out2_teacher['patches_after_head']], dim=0).flatten(0, 1)
                # i use first cls tokens of the second augmented signal to match the cls of the first augmented signal
                cls_tokens_stud = torch.cat([out2['cls_after_head'], out['cls_after_head']], dim=0)
                cls_tokens_teacher = torch.cat([out_teacher['cls_after_head'], out2_teacher['cls_after_head']], dim=0)

                patch_loss = embedding_cross_entropy_loss(stud_embeddings, teacher_embeddings, reduction='mean', mask=combined_mask, centering=self.centering, stud_temp=self.stud_temp, teacher_temp=self.teacher_temp)
                cross_cls_loss = embedding_cross_entropy_loss(cls_tokens_stud, cls_tokens_teacher, reduction='mean', centering=self.centering, stud_temp=self.stud_temp, teacher_temp=self.teacher_temp)

            self.log(f"{step}_patch_loss", patch_loss.item(), prog_bar=True, batch_size=batch_size)
            self.log(f"{step}_cross_cls_loss", cross_cls_loss.item(), prog_bar=True, batch_size=batch_size)

            teacher_student_loss = patch_loss + cross_cls_loss

            if self.use_sim_dino:
                cls_tok_stud_g = torch.stack([out['cls'], out2['cls']], dim=0)
                cls_tok_teacher_g = torch.stack([out_teacher['cls'], out2_teacher['cls']], dim=0)
            
                R_eps, compression_term, expansion_term = self.mcr_loss(cls_tok_stud_g, cls_tok_teacher_g)
                self.log(f"{step}_compression_term", compression_term.item(), prog_bar=False, batch_size=batch_size)
                self.log(f"{step}_expansion_term", expansion_term.item(), prog_bar=False, batch_size=batch_size)
                self.log(f"{step}_R_eps", R_eps.item(), prog_bar=True, batch_size=batch_size)

                teacher_student_loss = teacher_student_loss + R_eps
            
            if self.use_koleo_regularization:
                koleo_loss1 = self.koleo_reg(out['cls']) # on cls tokens
                koleo_loss2 = self.koleo_reg(out2['cls'])
                koleo_loss = (koleo_loss1 + koleo_loss2) / 2
                teacher_student_loss = teacher_student_loss + koleo_loss * 0.1
                self.log(f"{step}_koleo_loss", koleo_loss.item(), prog_bar=True, batch_size=batch_size)

            self.log(f"{step}_dino_loss", teacher_student_loss.item(), prog_bar=True, batch_size=batch_size)


            rank_me1 = self.rank_me(out['cls'])
            rank_me2 = self.rank_me(out2['cls'])
            self.log(f"{step}_rank_me", ((rank_me1 + rank_me2) / 2).item(), prog_bar=True, batch_size=batch_size)

            # log norm of output
            with torch.no_grad():
                if self.use_sim_dino:
                    norm = torch.norm(out['patches'], dim=-1)
                else: 
                    norm = torch.norm(out['patches_after_head'], dim=-1)
                norm = norm.mean()
                self.log(f"{step}_norm_emb", norm.item(), prog_bar=False, batch_size=self.batch_size)

                # log the mean cosine similarity between all samples in the batch
                cos_sim = torch.nn.functional.cosine_similarity(out['cls'].unsqueeze(1), out['cls'].unsqueeze(0), dim=-1)
                cos_sim2 = torch.nn.functional.cosine_similarity(out2['cls'].unsqueeze(1), out2['cls'].unsqueeze(0), dim=-1)
                cos_sim = (cos_sim.mean() + cos_sim2.mean()) / 2
                self.log(f"{step}_cos_sim", cos_sim.item(), prog_bar=False, batch_size=batch_size)
        
        if self.pretraining_strategy == 'masked_token_prediction' and self.model.use_teacher_student:
            nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(out['reconstruction'], x, mask = None) #  out['mask'])
            nrmse2, mse2, mae2, grad2, min_max2 = self.calculate_metrics_reconstruction(out2['reconstruction'], x2, mask=None) #, out2['mask'])
            nrmse, mse, mae, grad, min_max = (nrmse + nrmse2) / 2, (mse + mse2) / 2, (mae + mae2) / 2, (grad + grad2) / 2, (min_max + min_max2) / 2
        else:
            nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(out['reconstruction'], x, out['mask'])

        loss = torch.tensor(0.0, device=self.device)

        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse
        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True, batch_size=batch_size)
        self.log(f"{step}_mse", mse.item(), prog_bar=False, batch_size=batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=False, batch_size=batch_size)
        self.log(f"{step}_grad", grad.item(), prog_bar=False, batch_size=batch_size)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False, batch_size=batch_size)
        
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=False, batch_size=batch_size)

        return {'reconstruction_loss': loss, 'teacher_student_loss': teacher_student_loss}
    
    @torch.no_grad()
    def rank_me(self, tensor, eps=1e-8):
        try:
            _, S, _ = torch.svd(tensor)  # shape: (min(N, D),)

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
    

    def pad(self, x):
        # remove thte exceeding part of the signal not patchable
        if len(x.shape) == 2: x = x.unsqueeze(-1)
        part_to_remove = x.shape[1] % self.patch_size
        if part_to_remove != 0:
            x = x[:, :-part_to_remove, :]
        return x
    
    def get_params(self):
        return self.model.trainable_parameters()
    
    def get_lr(self):
        return self.lr

    def configure_optimizers(self):
        if self.model.use_teacher_student:
            return common.configure_optimizer_teacher_student(self)
        else:
            return common.configure_optimizers(self)

