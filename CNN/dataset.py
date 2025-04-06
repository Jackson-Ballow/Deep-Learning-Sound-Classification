import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, target_width=200, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.target_width = target_width

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = np.load(row['npy_path'])
        spec = self._pad_or_crop(spec, self.target_width)
        spec = np.expand_dims(spec, axis=0)

        if self.transform:
            spec = self.transform(spec)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(row['gender_label'], dtype=torch.long)

    def _pad_or_crop(self, spec, target_width):
        h, w = spec.shape
        if w < target_width:
            pad_w = target_width - w
            return np.pad(spec, ((0, 0), (0, pad_w)), mode='constant')
        elif w > target_width:
            return spec[:, :target_width]
        return spec
