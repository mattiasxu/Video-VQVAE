import torch
from dataset import DrivingDataset
from vidvqvae import VQVAE
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


dataset = DrivingDataset("./generative", frames=16, skip=16)
print(len(dataset))
train_set, val_set = random_split(dataset, [10000, 3009], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=16, num_workers=12)
val_loader = DataLoader(val_set, batch_size=8, num_workers=12)

model = VQVAE(
    in_channel=3,
    channel=128,
    n_res_block=2,
    n_res_channel=32,
    embed_dim=64,
    n_embed=512,
    decay=0.99
)

# wandb_logger = WandbLogger(project="VidVQVAE", log_model="all")
# wandb_logger.watch(model)

# checkpoint_callback = ModelCheckpoint(monitor="val_loss")
# trainer =  pl.Trainer(gpus=1, logger=wandb_logger, callbacks=[checkpoint_callback])
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)