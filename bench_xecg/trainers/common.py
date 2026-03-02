import numpy as np
from torch import optim
from bench_xecg.schedulers import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from typing_extensions import override

def configure_optimizers(trainer):
    if trainer.optimizer == 'adam':
        optimizer = optim.Adam(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
    elif trainer.optimizer == 'adamw':
        optimizer = optim.AdamW(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
    elif trainer.optimizer == 'adafactor':
        optimizer = optim.Adafactor(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
    elif trainer.optimizer == 'momentum':
        optimizer = optim.SGD(trainer.get_params(), lr=trainer.get_lr(), momentum=0.9, weight_decay=trainer.wd)
    elif trainer.optimizer == 'sgd':
        optimizer = optim.SGD(trainer.get_params(), lr=trainer.get_lr(), momentum=0., weight_decay=trainer.wd)

    if trainer.use_scheduler: 
        steps_per_epoch = np.ceil(trainer.len_train_dataset / trainer.batch_size)
        num_training_steps = steps_per_epoch * trainer.epochs
        warmup_steps = steps_per_epoch * trainer.num_epochs_warmup

        sched = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps = warmup_steps, 
            num_training_steps = num_training_steps, 
        )

        scheduler = {
            'scheduler': sched,
            'interval': 'step', # or 'epoch' 
            'frequency': 1,
        }
        return [optimizer], [scheduler]
    else:
        return [optimizer]
    
def configure_optimizer_teacher_student(trainer):
    if trainer.optimizer == 'adam':
        optimizer1 = optim.Adam(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
        optimizer2 = optim.Adam(params=trainer.model.reconstruction.parameters(), lr=trainer.get_reconstruction_lr(), weight_decay=trainer.wd)
    elif trainer.optimizer == 'adamw':
        optimizer1 = optim.AdamW(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
        optimizer2 = optim.AdamW(params=trainer.model.reconstruction.parameters(), lr=trainer.get_reconstruction_lr(), weight_decay=trainer.wd)
    elif trainer.optimizer == 'adafactor':
        optimizer1 = optim.Adafactor(params=trainer.get_params(), lr=trainer.get_lr(), weight_decay=trainer.wd)
        optimizer2 = optim.Adafactor(params=trainer.model.reconstruction.parameters(), lr=trainer.get_reconstruction_lr(), weight_decay=trainer.wd)
    else:
        optimizer1 = optim.SGD(trainer.get_params(), lr=trainer.get_lr(), momentum=0.9, weight_decay=trainer.wd)
        optimizer2 = optim.SGD(trainer.model.reconstruction.parameters(), lr=trainer.get_reconstruction_lr(), momentum=0.9, weight_decay=trainer.wd)

    if trainer.use_scheduler: 
        steps_per_epoch = np.ceil(trainer.len_train_dataset / trainer.batch_size)
        num_training_steps = steps_per_epoch * trainer.epochs
        warmup_steps = steps_per_epoch * trainer.num_epochs_warmup

        sched1 = get_cosine_schedule_with_warmup(
            optimizer1, 
            num_warmup_steps = warmup_steps, 
            num_training_steps = num_training_steps, 
        )

        sched2 = get_cosine_schedule_with_warmup(
            optimizer2, 
            num_warmup_steps = warmup_steps, 
            num_training_steps = num_training_steps, 
        )

        scheduler1 = {
            'scheduler': sched1,
            'interval': 'step', # or 'epoch' 
            'frequency': 1,
        }
        scheduler2 = {
            'scheduler': sched2,
            'interval': 'step', # or 'epoch' 
            'frequency': 1,
        }
        return [optimizer1, optimizer2], [scheduler1, scheduler2]
    else:
        return [optimizer1, optimizer2]

    

class DelayedCheckpoint(ModelCheckpoint):
    def __init__(self, delay_epochs=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_epochs = delay_epochs

    @override
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.delay_epochs:
            super().on_train_epoch_end(trainer, pl_module)