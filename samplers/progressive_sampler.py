import numpy as np
from torch.utils.data import Sampler


class ProgressiveSampler(Sampler):
    """
    Custom sampler for progressive sampling based on class imbalance ratios.
    """

    def __init__(self, labels, start_ratio=0.1, end_ratio=0.5, num_epochs=100):
        """
        Initialize the ProgressiveSampler with the provided labels, start_ratio, end_ratio, and num_epochs.
        Parameters:
        - labels: List of labels for the dataset.
        - start_ratio: Starting ratio for class imbalance.
        - end_ratio: Ending ratio for class imbalance.
        - num_epochs: Total number of epochs for training.
        """
        self.labels = labels
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_epochs = num_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * (self.epoch / self.num_epochs)
        indices = self.get_indices_based_on_ratio(ratio)
        return iter(indices)

    def get_indices_based_on_ratio(self, ratio):
        majority_indices = [i for i, label in enumerate(self.labels) if label == 1]
        minority_indices = [i for i, label in enumerate(self.labels) if label == 0]

        sampled_minority_indices = np.random.choice(minority_indices, int(len(majority_indices) * ratio), replace=True)
        return np.concatenate([majority_indices, sampled_minority_indices])

    def __len__(self):
        return len(self.labels)
