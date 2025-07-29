import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
import torch
from torch.utils.data.dataset import Dataset


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None, normalized=False):
        digits = load_digits()
        self.full_data = digits.data.astype(np.float32)  # for stat computation
        self.transforms = transforms
        self.normalized = normalized

        # Predefined split
        if mode == 'train':
            self.data = self.full_data[:1000]
        elif mode == 'val':
            self.data = self.full_data[1000:1350]
        else:
            self.data = self.full_data[1350:]

        # Compute normalization stats (from full dataset)
        if self.normalized:
            self.mean = torch.tensor(self.full_data[:1000].mean()).float()
            self.std = torch.tensor(self.full_data[:1000].std()).float()
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()

        if self.normalized:
            sample = (sample - self.mean) / self.std

        sample = sample.view(1,8,8)

        if self.transforms:
            sample = self.transforms(sample)

        return sample 


def plot_digits(batch, title="Digits", plot=True):
    batch = batch.cpu().numpy() if isinstance(batch, torch.Tensor) else batch

    n = len(batch)
    cols = min(10, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(1.5 * cols, 1.5 * rows))
    plt.title(title)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(batch[i,0], cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()

    if plot:
        plt.show()
    
    return fig