import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from models.vidvqvae import VQVAE

class LITVQVAE(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        self.model = VQVAE(
            in_channel=3,
            channel=params.channel,
            n_res_block=params.n_res_block,
            n_res_channel=params.n_res_channel,
            embed_dim=params.embed_dim,
            n_embed=params.n_embed
        )
        self.lr = params.lr
        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self.find_loss(train_batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.find_loss(val_batch, batch_idx)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        loss = self.find_loss(val_batch, batch_idx)
        self.log('test_loss', loss)

    def find_loss(self, batch, idx):
        x = y = batch
        x_hat, latent_loss = self.forward(x)
        recon_loss = self.criterion(x_hat, y)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss * self.latent_loss_weight
        # return loss
        return recon_loss

    def forward(self, input):
        dec, diff = self.model(input)
        return dec, diff


