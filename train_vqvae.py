import torch
from dataset import DrivingDataset
from vidvqvae import VQVAE
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


dataset = DrivingDataset("./tensor_data")
train_set, val_set = random_split(dataset, [20000, 6016], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=8, num_workers=12)
val_loader = DataLoader(val_set, batch_size=8, num_workers=12)

model = VQVAE(
    in_channel=3,
    channel=16
)

wandb_logger = WandbLogger(project="VidVQVAE", log_model="all")
trainer =  pl.Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)
# wandb_logger.watch(model)