
from pretrain import pretrain
import train_mit_bih as mit_bih
import train_ptb_xl as ptb_xl
import lightning as L
import wandb

def main():
    run = wandb.init()

    if wandb.config.task == 'pretrain':
        pretrain(run.config, run, wandb=True)
    elif wandb.config.task == 'mit_bih':
        mit_bih.train(run.config, run, wandb=True)
    elif wandb.config.task == 'ptb_xl':
        ptb_xl.train(run.config, run, wandb=True)
    else:
        raise ValueError("Task not supported")

main()