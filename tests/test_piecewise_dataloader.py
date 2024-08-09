import pytest
import torch
from torch.utils.data import DataLoader

from spd.scripts.piecewise.piecewise_dataset import PiecewiseDataset


def test_piecewise_dataset():
    # Define test parameters
    n_inputs = 5
    functions = [lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: torch.sin(x)]
    feature_probability = 0.3
    range_min = 0
    range_max = 5
    batch_size = 1000

    # Create dataset
    dataset = PiecewiseDataset(n_inputs, functions, feature_probability, range_min, range_max)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Get a batch of samples
    batch_x, batch_y = next(iter(dataloader))

    # Check shape
    assert batch_x.shape == (batch_size, n_inputs)
    assert batch_y.shape == (batch_size, 1)

    # Check first column (real values)
    assert torch.all((batch_x[:, 0] >= 0) & (batch_x[:, 0] <= range_max))

    # Check control bits (all but the last sample)
    control_bits = batch_x[:-1, 1:]
    assert torch.all((control_bits == 0) | (control_bits == 1))

    # Check mean of control bits
    mean_control_bits = control_bits.float().mean()
    assert pytest.approx(mean_control_bits.item(), abs=0.05) == feature_probability
