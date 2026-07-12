# dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as f:
            self.data = f['data'][:]  # [N, C, H, W]，原样保留！
        self.transform = transform
        print(f"Loaded data shape: {self.data.shape}")  # 确认是 (N, 2, 64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.data[idx])  # [C, H, W]，C=2 (例如vis/ir或oe/ue)
        return img