import torch
from tqdm import tqdm
import os
from PIL import ImageFile
import PIL
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import namedtuple
import lmdb
import pickle
from torch.nn.functional import one_hot


ImageFile.LOAD_TRUNCATED_IMAGES = True
PATH = "./tensor_data"

Data = namedtuple('Data', ['top', 'bottom'])


class LatentDataset(Dataset):
    def __init__(self, path, n_embed=512):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.n_embed = n_embed

        if not self.env:
            raise IOError('Cannot open dataset', path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')
            data = pickle.loads(txn.get(key))
        top = one_hot(torch.from_numpy(data.top), self.n_embed).permute(3, 0, 1, 2).type(torch.FloatTensor)
        bot = one_hot(torch.from_numpy(data.bottom), self.n_embed).permute(3, 0, 1, 2).type(torch.FloatTensor)
        return top, bot


class DrivingDataset(Dataset):
    def __init__(self, path, frames=16, skip=8):
        self.path = path
        self.data = os.listdir(path)
        self.frames = frames
        self.skip = skip
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def to_tensor(self, idx):
        img = PIL.Image.open(
            self.path + f"/{idx}.png"
        ).convert('RGB')
        return self.transforms(img)

    def __len__(self):
        return 1 + (len(self.data) - self.frames) // self.skip

    def __getitem__(self, idx):
        frames = []
        for i in range(idx * self.skip, idx * self.skip + self.frames):
            frames.append(self.to_tensor(i))
        return torch.stack(frames, 1)


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

    test = DrivingDataset("./generative", frames=16, skip=8)
    print(len(test))
    train_set, val_set = torch.utils.data.random_split(test, [20000, 6017])
    print(len(train_set))
