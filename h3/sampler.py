"""
Information-weighted sampler for H3 optimizer.
"""

import torch
from torch.utils.data import Sampler


class InformationWeightedSampler(Sampler):
    """
    Sampler that prioritizes training samples based on their information content.

    Uses gradient magnitude and loss values to compute sample importance weights.
    """

    def __init__(self, dataset, batch_size=32):
        """
        Initialize information-weighted sampler.

        Args:
            dataset: Dataset to sample from
            batch_size: Batch size for sampling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.weights = torch.ones(self.num_samples)

    def update_weights(self, indices, losses, gradients):
        """
        Update sample weights based on losses and gradient information.

        Args:
            indices: Sample indices
            losses: Loss values for samples
            gradients: Gradient magnitudes for samples
        """
        # TODO: Implement information-based weight update
        pass

    def __iter__(self):
        """Generate sampling indices based on information weights."""
        # TODO: Implement weighted sampling logic
        return iter(torch.randperm(self.num_samples).tolist())

    def __len__(self):
        """Return the number of samples."""
        return self.num_samples
