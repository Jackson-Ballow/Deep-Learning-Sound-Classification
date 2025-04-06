import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, target_width=128, task='gender'):
        self.df = pd.read_csv(csv_path)
        self.target_width = target_width
        self.task = task

        if task == 'gender':
            self.label_column = 'gender_label'
        elif task == 'age':
            self.label_column = 'age_label'
        else:
            raise ValueError("Task must be 'gender' or 'age'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['npy_path']
        label = int(row[self.label_column])

        spec = np.load(path) 
        
        # Pad or crop to fixed width
        if spec.shape[1] < self.target_width:
            pad_width = self.target_width - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        elif spec.shape[1] > self.target_width:
            spec = spec[:, :self.target_width]

        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        return tensor, label
