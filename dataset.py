import torch
import os
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    def __init__(self, path, frames=16):
        self.path = path
        self.frames = frames
        self.imgs = os.listdir(path)
    
    def __len__(self):
        return len(self.imgs) - self.frames + 1