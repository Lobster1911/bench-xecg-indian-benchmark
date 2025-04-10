
from pretrain import pretrain
import train_mit_bih as mit_bih
import train_ptb_xl as ptb_xl
import lightning as L
import wandb
from utils.utils import parse_sweep_config

def main():
    run = wandb.init()

    config = parse_sweep_config(run.config, 'configs/train_ptb_xl_config_defaults.yaml')


    if wandb.config.task == 'pretrain':
        pretrain(config, run, wandb=True)
    elif wandb.config.task == 'mit_bih':
        mit_bih.train(config, run, wandb=True)
    elif wandb.config.task == 'ptb_xl':
        ptb_xl.train(config, run, wandb=True)
    else:
        raise ValueError("Task not supported")

main()