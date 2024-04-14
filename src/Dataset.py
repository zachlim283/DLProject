import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type, dataset_ver):
        self.X = np.array(np.load(f'Generated_Datasets/{dataset_type}_data_{dataset_ver}.npy'))
        self.y = np.array(np.load(f'Generated_Datasets/{dataset_type}_labels_{dataset_ver}.npy'))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).float()
        y = torch.from_numpy(np.asarray(self.y[idx])).float()
        return X, y