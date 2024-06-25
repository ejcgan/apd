import pytest
import torch
import numpy as np
from spd.scripts.tms_decomposition_highdim import calculate_closeness_to_identity

def test_calculate_closeness_to_identity():
    # Test case 1: Perfect identity matrix
    A = torch.eye(5, 3).unsqueeze(0)  # Shape: (1, 5, 3)
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 2: Matrix close to identity
    A = torch.eye(5, 3).unsqueeze(0) + 0.1 * torch.randn(1, 5, 3)
    closeness = calculate_closeness_to_identity(A)
    assert 0.8 < closeness < 1.0

    # Test case 3: Random matrix
    torch.manual_seed(42)
    A = torch.randn(1, 5, 3)
    closeness = calculate_closeness_to_identity(A)
    assert 0.0 < closeness < 0.8

    # Test case 4: Multiple instances
    A = torch.stack([torch.eye(5, 3), torch.randn(5, 3)])
    closeness = calculate_closeness_to_identity(A)
    assert 0.5 < closeness < 1.0

    # Test case 5: More features than k
    A = torch.eye(7, 5).unsqueeze(0)
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 6: More k than features
    A = torch.eye(5, 7).unsqueeze(0)
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(1.0, abs=1e-6)

def test_calculate_closeness_to_identity_edge_cases():
    # Test case 7: Empty matrix
    with pytest.raises(ValueError):
        A = torch.empty(1, 0, 0)
        calculate_closeness_to_identity(A)

    # Test case 8: 1x1 matrix
    A = torch.ones(1, 1, 1)
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(0.5, abs=1e-6)

    # Test case 9: Matrix with negative values
    A = torch.tensor([[[-1, 0], [0, 1]]]).float()
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 10: Matrix with very large values
    A = 1e6 * torch.eye(3, 3).unsqueeze(0)
    closeness = calculate_closeness_to_identity(A)
    assert closeness == pytest.approx(1.0, abs=1e-6)
