import argparse

import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset import LatentDataset
from pixelsnail import HierarchicalPixelSNAIL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = LatentDataset(args.path, n_embed=512)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=12)

    model = HierarchicalPixelSNAIL(512, 64, 2, 2, 2, 16, 128)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, loader)
