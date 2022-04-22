import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from lit_topsnail import TopSNAIL
from dataset import LatentDataModule

hyperparameter_defaults = dict(
        path='./latent_data',
        lr=0.001,
        batch_size=2,
        epochs=1,
        input_channels=512,
        n_codes=512,
        n_snail_blocks=6,
        key_channels=16,
        value_channels=128,
        n_filters=256,
        n_res_blocks=1
)

wandb.init(config=hyperparameter_defaults)
config = wandb.config

def main(config):
    topsnail = TopSNAIL(config)
    latent_data = LatentDataModule(config)
    # wandb_logger = WandbLogger()
    # wandb_logger.watch(topsnail)

    trainer = pl.Trainer(
            gpus=1,
            max_epochs=config.epochs,
            fast_dev_run=True,
            )
    trainer.fit(topsnail, latent_data)

if __name__ == "__main__":
    main(config)
