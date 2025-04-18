import lightning as L
from utils.train_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss, embedding_cross_entropy_loss, vicreg_loss
from torch.nn import functional as F
from utils.plot_utils import plot_reconstruction, plot_generation
import numpy as np
import torch
import lightning
import trainers.common as common
import torch.distributed


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
        self.use_vic_reg_regularization = config.use_vic_reg_regularization
        self.beta_std = config.beta_std
        self.beta_cov = config.beta_cov

        if self.model.use_teacher_student:
            self.automatic_optimization=False

        if not config.is_sweep:
            self.save_hyperparameters()

    def training_step(self, batch, _):
        if self.model.use_teacher_student:
            head_loss, jepa_loss = self.reconstruct_batch(batch, step='train')
            opt_core, opt_head = self.optimizers()
            sched_core, sched_head = self.lr_schedulers()

            train_head = self.current_epoch >= self.start_train_head_at_epoch
            
            opt_core.zero_grad()
            self.manual_backward(jepa_loss, retain_graph=train_head)
            opt_core.step()
            sched_core.step()

            if train_head:
                opt_head.zero_grad()
                self.manual_backward(head_loss)
                opt_head.step()
                sched_head.step()

            self.update_teacher()
            return
        else:
            loss = self.reconstruct_batch(batch, step='train')
            return loss
    
    def update_teacher(self):
        with torch.no_grad():
            steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
            num_training_steps = steps_per_epoch * self.epochs
            beta = self.model.ema_0 + self.global_step * (self.model.ema_1 - self.model.ema_0) / num_training_steps

            # the xlstm teacher is present only if the strategy is multi token prediction
            if self.pretraining_strategy == 'masked_token_prediction':
                for param_s, param_t in zip(self.model.xlstm.parameters(), self.model._xlstm_teacher.parameters()):
                    param_t.data = param_t.data * beta + (1.0 - beta) * param_s.data

            for param_s, param_t in zip(self.model.patch_embedding.parameters(), self.model._patch_embedding_teacher.parameters()):
                param_t.data = param_t.data * beta + (1.0 - beta) * param_s.data

            self.model._vocab_teacher.weight.data = self.model._vocab_teacher.weight.data * beta + (1.0 - beta) * self.model.vocab.weight.data
        
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

        if self.pretraining_strategy == 'next_token_prediction':
            x, reconstruction, out_teacher, last_emb = self.next_token_prediction(batch)
        if self.pretraining_strategy == 'masked_token_prediction':
            # masking is automatically done inside this function
            # x will be masked with the inverse of the mask
            # so the loss  function will automatically skip the masked values
            x, reconstruction, out_teacher, last_emb = self.masked_token_prediction(batch)

        nrmse = np.inf

        # compute the loss and use the gradients only when it is needed
        if 'min_max' in self.loss_type:
            min_max = masked_min_max_loss(reconstruction, x, patch_size=self.patch_size)

        if 'mae' in self.loss_type:
            mae = masked_mae_loss(reconstruction, x)
        else:
            with torch.no_grad(): mae = masked_mae_loss(reconstruction, x)

        if 'grad' in self.loss_type:
            grad = gradient_loss(reconstruction, x)
        else:
            with torch.no_grad(): grad = gradient_loss(reconstruction, x)
        
        if 'mse' in self.loss_type:
            mse = masked_mse_loss(reconstruction, x, reduction='mean')
        else:
            with torch.no_grad(): mse = masked_mse_loss(reconstruction, x, reduction='mean', mask= x != 0)
   
        # calculate the normalized root squared error only for the first token prediction
        with torch.no_grad():
            nrmse = torch.sqrt(mse) / (x.max() - x.min())
        
        loss = torch.tensor(0.0, device=self.device)
        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse

        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)

        self.log(f"{step}_mse", mse.item(), prog_bar=False, batch_size=self.batch_size)
        self.log(f"{step}_mae", mae.item(), prog_bar=False, batch_size=self.batch_size)
        self.log(f"{step}_grad", grad.item(), prog_bar=False, batch_size=self.batch_size)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False, batch_size=self.batch_size)
        
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=True, batch_size=self.batch_size)

        if self.model.use_teacher_student:
            # last_emb [bs, seq_len -1, num_hiddens]
            # x [bs, seq_len * patch_size, channels]
            bs, seq_len, channels = x.shape
            x_reshaped = x.view(bs, seq_len // self.patch_size, self.patch_size, channels)
            non_zero_mask = x_reshaped.abs().sum(dim=(2, 3)) > 0  # shape: [bs, seq_len]


            if self.pretraining_strategy == 'next_token_prediction':
                teacher_student_loss = embedding_cross_entropy_loss(last_emb, out_teacher, mask=non_zero_mask, reduction='mean')
            if self.pretraining_strategy == 'masked_token_prediction':
                teacher_student_loss = masked_mse_loss(last_emb, out_teacher, reduction='mean', mask=non_zero_mask)

            if self.use_vic_reg_regularization:
                std_loss, cov_loss = vicreg_loss(last_emb)
                self.log(f"{step}_std_loss", std_loss.item(), prog_bar=False, batch_size=self.batch_size)
                self.log(f"{step}_cov_loss", cov_loss.item(), prog_bar=False, batch_size=self.batch_size)
                self.log(f"{step}_teacher_student_loss", teacher_student_loss.item(), prog_bar=False, batch_size=self.batch_size)
                teacher_student_loss = teacher_student_loss + std_loss * self.beta_std + cov_loss * self.beta_cov

            self.log(f"{step}_jepa_loss", teacher_student_loss.item(), prog_bar=True, batch_size=self.batch_size)

            return loss, teacher_student_loss

        return loss

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
    
    def masked_token_prediction(self, batch):
        x = self.pad(batch["signal"])
        mask = self.get_random_mask(x)

        # mask a rnadom number of patches
        masked_x = x.masked_fill(~mask, 0)
        reconstruction, out_teacher, last_emb = self.model(masked_x)
        inverted_masked_x = x.masked_fill(mask, 0)

        if self.model.use_teacher_student:
            mask = mask.view(mask.shape[0], out_teacher.shape[1], self.patch_size).sum(dim=-1) == 0
            mask = mask.unsqueeze(-1)
            out_teacher = out_teacher.masked_fill(mask, 0)
            return inverted_masked_x, reconstruction, out_teacher, last_emb

        # needed for the loss function, if the masked value is 0, then the loss function will not consider it
        return inverted_masked_x, reconstruction, None, None
    
    def get_random_mask(self, x):
        # masking the signal
        num_patches = x.shape[1] // self.patch_size
        rand = torch.rand(x.shape[0], num_patches, device=self.device)
        mask = (rand > self.mask_ratio) # this is true for non masked
        # repeat the mask to num_patches * patch_size
        mask = mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(-1)
        return mask

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

