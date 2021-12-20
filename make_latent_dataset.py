import argparse
import pickle
from tqdm import tqdm
import lmdb
import torch
from torch.utils.data import DataLoader
from dataset import DrivingDataset, Data
from vidvqvae import VQVAE


def extract(lmdb_env, dataloader, model, device):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        dataloader = tqdm(dataloader)

        for vid in dataloader:
            vid = vid.to(device)

            _, _, _, id_t, id_b = model.encode(vid)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for top, bottom in zip(id_t, id_b):
                data = Data(top=top, bottom=bottom)
                txn.put(str(index).encode('utf-8'), pickle.dumps(data))
                index += 1
                dataloader.set_description(f"Inserted: {index}")

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('checkpoint', type=str)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = DrivingDataset(args.path, frames=16, skip=8)
    loader = DataLoader(dataset, batch_size=8, num_workers=12, shuffle=False)

    model = VQVAE.load_from_checkpoint(args.checkpoint, in_channel=3, channel=16)
    model.to(device)
    model.eval()

    env = lmdb.open("test", map_size=100*1024*1024*1024)
    extract(env, loader, model, device)
    print("Success")
