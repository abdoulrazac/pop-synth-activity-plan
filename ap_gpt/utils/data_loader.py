import torch
import numpy as np


class DataLoader(torch.utils.data.Dataset):
    """
    Custom DataLoader class to load data from a pandas DataFrame.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray = None, batch_size: int = 32, shuffle: bool = True):
        # If only X is provided and it's a tuple, assume it contains both X and y
        if y is None and isinstance(X, tuple) and len(X) == 2:
            self.X, self.y = X
        else:
            self.X, self.y = X, y

        # Convert to tensors if needed
        if not isinstance(self.X, torch.Tensor):
            self.X = torch.tensor(self.X, dtype=torch.float32)

        if self.y is not None and not isinstance(self.y, torch.Tensor):
            self.y = torch.tensor(self.y, dtype=torch.long)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(self.X)
        self.indices = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_idx = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = self.indices[idx]
        else:
            indices = [self.indices[idx]]

        X_batch = self.X[indices]

        if self.y is not None:
            y_batch = self.y[indices]
            return X_batch, y_batch
        else:
            return X_batch

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        self.current_idx = end_idx

        X_batch = self.X[batch_indices]

        if self.y is not None:
            y_batch = self.y[batch_indices]
            return X_batch, y_batch
        else:
            return X_batch