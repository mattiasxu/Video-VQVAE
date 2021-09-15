import torch
from tqdm import tqdm
import os
import PIL
from torch.utils.data import Dataset
import torchvision.transforms as transforms

PATH = "./tensor_data"

class DrivingDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = os.listdir(path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.load(self.path + f"{idx}.pt")

class Img2VidTensor():
    def __init__(self, path, save_dir, frames, skip):
        self.path = path
        self.save_dir = save_dir
        self.frames = frames
        self.skip = skip
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def to_tensor(self, idx):
        img = PIL.Image.open(
            self.path + f"/{idx}" + ".png"
        ).convert('RGB')
        return self.transforms(img)
    
    def videofy(self, idx):
        vid = torch.zeros(3, self.frames, 256, 256)
        time = 0
        for i in range(idx*self.skip, idx*self.skip + self.frames):
            img = self.to_tensor(i)
            vid[:, time, :, :] = img
            time += 1
        torch.save(vid, f"{self.save_dir}/{idx}.pt")
        return vid
    
class RunningStats():
    def __init__(self):
        self.n = 0
        self.old_m = torch.Tensor([0., 0., 0.])
        self.new_m = torch.Tensor([0., 0., 0.])
        self.old_s = torch.Tensor([0., 0., 0.])
        self.new_s = torch.Tensor([0., 0., 0.])
    
    def push(self, x):
        self.n += 1
        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = torch.Tensor([0., 0., 0.])
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            
            self.old_m = self.new_m
            self.old_s = self.new_s
    
    def mean(self):
        return self.new_m if self.n else torch.Tensor([0., 0., 0.])
        
    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else torch.Tensor([0., 0., 0.])

    def std(self):
        return torch.sqrt(self.variance())

if __name__ == "__main__":
    """
    stats = RunningStats()
    for file in tqdm(os.listdir("./tensor_data/")):
        x = torch.load("./tensor_data/" + file)
        for time in range(0, 16):
            for pixels in x[:, time, :, :].reshape(256*256, 3):
                stats.push(pixels)
        print(stats.std())
    """
    
    test = DrivingDataset("./tensor_data")
    print(len(test))
    train_set, val_set = torch.utils.data.random_split(test, [20000, 6016])
    print(len(train_set))