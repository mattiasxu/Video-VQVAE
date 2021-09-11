import torch
import os
import PIL
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    def __init__(self, path, frames=16):
        self.path = path
        self.frames = frames
        self.imgs = os.listdir(path)
        self.transform = None
    
    def __len__(self):
        return len(self.imgs)//8 - 1
    
    def __getitem__(self, idx):
        video = torch.zeros(3, 16, 256, 256)
        time = 0
        for i in range(idx*8, idx*8+16):
            img = PIL.Image.open(
                self.path + f"/{idx*8}" + ".jpg"
            )
            img = self.transform(img)
            video[:, time, :, :] = img 
        return video