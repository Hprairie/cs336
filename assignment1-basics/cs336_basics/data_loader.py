import numpy as np
import torch
import random

# Naive Implementation
def data_loader(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    start_range = len(data) - context_length
    data_points = np.random.randint(0, start_range, size=batch_size)
    x = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size, context_length))
    for i, start in enumerate(data_points):
        x[i] = data[start:start + context_length]
        y[i] = data[start+1:start + context_length+1]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device.type == "cuda":
        x = x.pin_memory().to(device=device, non_blocking=True)
        y = y.pin_memory().to(device=device, non_blocking=True)
    else:
        x = x.to(device=device)
        y = y.to(device=device)
    return x, y

# Semi Advanced Dataloading
class Sampler:
    def __init__(self, context_length, dataset_size) -> None:
        self.context_length = context_length
        self.dataset_size = dataset_size - 1
        self.sample_size = dataset_size // context_length
        self.remaining_samples = dataset_size % context_length
        self.refresh()
    
    def create_sample_order(self) -> np.ndarray:
        raise NotImplementedError
    
    def create_sample_offset(self) -> int:
        raise NotImplementedError

    def __call__(self, batch_size: int) -> np.ndarray:
        batch_size = max(batch_size, len(self.sample_order))
        vals = self.sample_order[:batch_size]
        self.sample_order = self.sample_order[batch_size:]
        return self.sample_offset + vals * self.sample_size

    def __len__(self) -> int:
        return len(self.sample_order)
    
    def refresh(self):
        self.sample_order = self.create_sample_order()
        self.sample_offset = self.create_sample_offset()


class RandomSampler(Sampler):
    def create_sample_order(self) -> np.ndarray:
        return np.random.permutation(self.sample_size)

    def create_sample_offset(self) -> int:
        return random.randint(0, self.remaining_samples)
    

class OrderedSampler(Sampler):
    def create_sample_order(self) -> np.ndarray:
        return np.arange(self.sample_size)

    def create_sample_offset(self) -> int:
        return 0
    
class DataLoader:
    def __init__(self, dataset: np.ndarray, sampler: Sampler, steps: int | None = None, batch_size: int = 1, context_length: int = 1, pin_memory: bool = False, drop_last: bool = False, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.steps = steps
        self.sampler = sampler
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        assert isinstance(device, torch.device), "Must be a valid torch device"
        self.device = device
        assert isinstance(dtype, torch.dtype), "Must be a valid torch dtype"
        self.dtype = dtype
    
    def to(self, device: torch.device, dtype: torch.dtype | None = None):
        assert isinstance(device, torch.device), "Must be a valid torch device"
        self.device = torch.device
        if dtype is not None:
            assert isinstance(dtype, torch.dtype), "Must be a valid torch dtype"
            self.dtype = dtype
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.steps
        if self.idx < self.steps:
            # Check is sampler still has available samples
            if not len(self.sampler):
                self.sampler.refresh()
            if len(self.sampler) < self.batch_size and self.drop_last:
                self.sampler.refresh()
            # Get samples start from the sampler
            samples = self.sampler(batch_size=self.batch_size)
            x = np.zeros((self.batch_size, self.context_length))
            y = np.zeros((self.batch_size, self.context_length))
            # Retrieve samples from the dataset
            for i, start in enumerate(samples):
                x[i] = self.dataset[start:start + self.context_length]
                y[i] = self.dataset[start+1:start + self.context_length+1]
            # Convert samples to torch tensors and move to correct device
            x = torch.from_numpy(x).to(dtype=self.dtype)
            y = torch.from_numpy(y).to(dtype=self.dtype)
            if self.device.type == "cuda":
                x = x.pin_memory().to(device=self.device, non_blocking=True)
                y = y.pin_memory().to(device=self.device, non_blocking=True)
            else:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
            return x, y
        else:
            raise StopIteration
    