import pytorch_lightning as pl
import wandb

from pytorch_lightning.loggers import WandbLogger
from lit_vidvqvae import LITVQVAE
from dataset import DataModule

hyperparameter_defaults = dict(
    path='./generative',
    lr=0.001,
    batch_size=2,
    n_embed=512,
    embed_dim=32,
    n_res_block=2,
    n_res_channel=128,
    channel=128,
    epochs=2
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

def main(config):
    vidvqvae = LITVQVAE(config)
    driving_data = DataModule(config)
    wandb_logger = WandbLogger()
    wandb_logger.watch(vidvqvae)

    trainer = pl.Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=config.epochs
    )
    trainer.validate(vidvqvae, driving_data)

if __name__ == "__main__":
    main(config)
