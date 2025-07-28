import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


def make_spiral(n=1000, rotations=2, noise=0.5, seed=None):
    t = np.linspace(0, 2*np.pi * rotations, n)
    X = np.column_stack((t * np.sin(t), t * np.cos(t)))
    
    rng = np.random.default_rng(seed)
    X += noise * rng.standard_normal((n, 2))
    
    return X, t


class GaussianSpiralDataset(Dataset):
    def __init__(self, n, rotations=2, var=0.5, seed=None, mode="train", mean=None, std=None):
        assert mode in {"train", "val", "test"}, "mode must be 'train', 'val', or 'test'"
        self.mode = mode

        data, self.t = make_spiral(n=n, rotations=rotations, noise=var, seed=seed)
        self.data = torch.from_numpy(data).float()

        if mode == "train":
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)
        else:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for val/test mode.")
            self.mean = mean
            self.std = std

        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.t[idx]

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std
  

def plot_gaussian_spiral(batch, t=None):
    batch = batch.cpu().numpy() if isinstance(batch, torch.Tensor) else batch

    if batch.shape[1] != 3:
        raise ValueError(f"Expected input with shape [N, 2], but got {batch.shape}")
    
    plt.figure(figsize=(6, 5))
    plt.scatter(batch[:, 0], batch[:, 1], c=t, s=2, alpha=0.6, cmap='viridis' if t is not None else None)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Gaussian Spiral")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()