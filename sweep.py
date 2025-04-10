
from pretrain import pretrain
import train_mit_bih as mit_bih
import train_ptb_xl as ptb_xl
import lightning as L
import wandb
from utils.utils import parse_sweep_config

def main():
    run = wandb.init()

    if wandb.config.task == 'pretrain':
        config = parse_sweep_config(run.config, 'config_defaults/pretrain_config_defaults.yaml')
        pretrain(config, run, wandb=True)
    elif wandb.config.task == 'mit_bih':
        config = parse_sweep_config(run.config, 'config_defaults/train_mit_bih_config_defaults.yaml')
        mit_bih.train(config, run, wandb=True)
    elif wandb.config.task == 'ptb_xl':
        config = parse_sweep_config(run.config, 'config_defaults/train_ptb_xl_config_defaults.yaml')
        ptb_xl.train(config, run, wandb=True)
    else:
        raise ValueError("Task not supported")

main()