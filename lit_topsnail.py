import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from models.pixelsnail import PixelSNAIL

class TopSNAIL(pl.LightningModule):
    def __init__(self, params):
        super().__init__()

        print(params.input_channels)
        self.model = PixelSNAIL(
                attention=True,
                input_channels=params.input_channels,
                n_codes=params.n_codes,
                n_snail_blocks=params.n_snail_blocks,
                key_channels=params.key_channels,
                value_channels=params.value_channels,
                n_filters=params.n_filters,
                n_res_blocks=params.n_res_blocks
                )

        self.lr = params.lr # params.lr
        self.criterion = nn.NLLLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = self.find_loss(train_batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.find_loss(val_batch, batch_idx)
        self.log('val_loss', loss)

    def find_loss(self, batch, idx):
        x = batch[0]
        y = torch.argmax(x, dim=1)
        print(x.shape)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, y)
        return loss

    def forward(self, top_code):
        return self.model(top_code)
