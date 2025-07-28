from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt

class SwissRollDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, mode="train", mean=None, std=None, dim="3D"):
        assert mode in {"train", "val", "test"}, "mode must be 'train', 'val', or 'test'"
        self.mode = mode
        self.dim = dim

        data, self.t = make_swiss_roll(n_samples=num_samples, noise=0.0)
        if dim != "3D": #2D projection
            data = data[:, [0, 2]]  # use X and Z axes
        self.data = torch.from_numpy(data).float()

        if mode == "train":
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)
        else:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for val/test mode.")
            self.mean = mean
            self.std = std

        # Normalize
        self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.t[idx]

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std
    

def plot_swiss_roll(batch, t=None):
    batch = batch.cpu().numpy() if isinstance(batch, torch.Tensor) else batch
    dim = batch.shape[1]

    if dim == 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(batch[:, 0], batch[:, 1], c=t, s=2, alpha=0.6, cmap='viridis' if t is not None else None)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("Swiss Roll (2D projection)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif dim == 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2], c=t, s=2, alpha=0.6, cmap='viridis' if t is not None else None)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Swiss Roll (3D)")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Expected input with shape [N, 2] or [N, 3], but got {batch.shape}")
