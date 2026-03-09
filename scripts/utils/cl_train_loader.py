import torch
from torch_geometric.loader import DataLoader
import argparse
import random
import yaml
import lmdb
import pickle
import zlib

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path1, lmdb_path2):
        self.env1 = lmdb.open(lmdb_path1, readonly=True, subdir=False, lock=False)
        self.env2 = lmdb.open(lmdb_path2, readonly=True, subdir=False, lock=False)
        with self.env1.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        assert len(self.keys) == self._get_num_keys(self.env2) 

    def _get_num_keys(self, env):
        with env.begin() as txn:
            return sum(1 for _ in txn.cursor())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env1.begin() as txn1, self.env2.begin() as txn2:
            g1 = pickle.loads(zlib.decompress(txn1.get(key)))
            g2 = pickle.loads(zlib.decompress(txn2.get(key)))
        return idx, g1, g2

def get_data_loaders(aug1_path, aug2_path, batch_size=32, shuffle=True, val_ratio=0.1, val_batch_size=None):
    dataset = ContrastiveDataset(aug1_path, aug2_path)
    total = len(dataset)
    indices = list(range(total))
    split = int(total * (1 - val_ratio))
    if shuffle:
        random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    if val_batch_size is None:
        val_batch_size = batch_size
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the YAML config file')
    parser.add_argument('--all_ids', type=str, default='./graph1/all_ids.npy')
    parser.add_argument('--aug1', type=str, default='./graph1/aug_graphs_1.lmdb')
    parser.add_argument('--aug2', type=str, default='./graph1/aug_graphs_2.lmdb')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_path', type=str, default='./outputs_cl')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for contrastive loss calculation')
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            yaml_args = yaml.safe_load(f)
        for key, value in yaml_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args
