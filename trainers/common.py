import numpy as np
from torch import optim
from schedulers import get_cosine_with_hard_restarts_schedule_with_warmup_and_decay

def configure_optimizers(trainer):
    if trainer.optimizer == 'adam':
        optimizer = optim.Adam(params=trainer.get_params(), lr=trainer.lr_head, weight_decay=trainer.wd)
    elif trainer.optimizer == 'adamw':
        optimizer = optim.AdamW(params=trainer.get_params(), lr=trainer.lr_head, weight_decay=trainer.wd)
    elif trainer.optimizer == 'adafactor':
        optimizer = optim.Adafactor(params=trainer.get_params(), lr=trainer.lr_head, weight_decay=trainer.wd)
    else:
        optimizer = optim.SGD(trainer.get_params(), lr=trainer.lr_head, momentum=0.9, weight_decay=trainer.wd)

    if trainer.use_scheduler: 
        steps_per_epoch = np.ceil(trainer.len_train_dataset / trainer.batch_size)
        num_training_steps = steps_per_epoch * trainer.epochs
        warmup_steps = steps_per_epoch * trainer.num_epochs_warmup

        sched = get_cosine_with_hard_restarts_schedule_with_warmup_and_decay(
            optimizer, 
            num_warmup_steps = warmup_steps, 
            num_training_steps = num_training_steps, 
            num_cycles = (num_training_steps // warmup_steps) // trainer.num_epochs_warm_restart,
            decay_factor=trainer.sched_decay_factor
        )

        scheduler = {
            'scheduler': sched,
            'interval': 'step', # or 'epoch' 
            'frequency': 1,
        }
        return [optimizer], [scheduler]
    else:
        return [optimizer]