
from pretrain import pretrain
from train import train
import lightning as L
import wandb

def main():
    run = wandb.init()
    if wandb.config.pretrain:
        pretrain(run.config, run, wandb=True)
    else:
        train(run.config, run, wandb=True)

main()