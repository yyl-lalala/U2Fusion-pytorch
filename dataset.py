# dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as f:
            self.data = f['data'][:]  # [N, C, H, W]
        self.transform = transform
        print(f"Loaded data shape: {self.data.shape}")  #  (N, 2, 64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.data[idx])  # [C, H, W]，C=2 
        return img
