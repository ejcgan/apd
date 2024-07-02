import pytest
import torch

from spd.scripts.tms_decomposition import calculate_closeness_to_identity


def test_calculate_closeness_to_identity():
    # Test case 1: Perfect identity matrix
    x = torch.eye(5, 3).unsqueeze(0)  # Shape: (1, 5, 3)
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 2: Matrix close to identity
    x = torch.eye(5, 3).unsqueeze(0) + 0.1 * torch.randn(1, 5, 3)
    closeness = calculate_closeness_to_identity(x)
    assert 0.8 < closeness < 1.0

    # Test case 3: Random matrix
    torch.manual_seed(42)
    x = torch.randn(1, 5, 3)
    closeness = calculate_closeness_to_identity(x)
    assert 0.0 < closeness < 0.8

    # Test case 4: Multiple instances
    x = torch.stack([torch.eye(5, 3), torch.randn(5, 3)])
    closeness = calculate_closeness_to_identity(x)
    assert 0.5 < closeness < 1.0

    # Test case 5: More features than k
    x = torch.eye(7, 5).unsqueeze(0)
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 6: More k than features
    x = torch.eye(5, 7).unsqueeze(0)
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(1.0, abs=1e-6)


def test_calculate_closeness_to_identity_edge_cases():
    # Test case 7: Empty matrix
    with pytest.raises(ValueError):
        x = torch.empty(1, 0, 0)
        calculate_closeness_to_identity(x)

    # Test case 8: 1x1 matrix
    x = torch.ones(1, 1, 1)
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(0.5, abs=1e-6)

    # Test case 9: Matrix with negative values
    x = torch.tensor([[[-1, 0], [0, 1]]]).float()
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(1.0, abs=1e-6)

    # Test case 10: Matrix with very large values
    x = 1e6 * torch.eye(3, 3).unsqueeze(0)
    closeness = calculate_closeness_to_identity(x)
    assert closeness == pytest.approx(1.0, abs=1e-6)
