import pytest
import torch

from spd.utils import calculate_closeness_to_identity, permute_to_identity


def test_permute_to_identity():
    # Test case 0: Identity matrix with some negative values
    A0 = torch.eye(3)
    A0[1] *= -1
    expected0 = torch.eye(3)

    # Test case 1: Square matrix (from the previous example)
    A1 = torch.tensor(
        [[0.1, 0.2, 0.9, 0.1], [0.8, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.9], [0.1, 0.9, 0.1, 0.1]]
    )
    expected1 = torch.tensor(
        [[0.8, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1], [0.1, 0.2, 0.9, 0.1], [0.1, 0.1, 0.1, 0.9]]
    )

    # Test case 2: Non-square matrix (more rows than columns)
    A2 = torch.tensor([[0.1, 0.9, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.1]])
    expected2 = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.8, 0.1, 0.1]])

    # Test case 3: Non-square matrix (more columns than rows)
    A3 = torch.tensor([[0.1, 0.2, 0.9, 0.1], [1.1, 0.1, 0.1, 0.1], [0.1, 1.0, 0.1, 0.001]])
    expected3 = torch.tensor([[1.1, 0.1, 0.1, 0.1], [0.1, 1.0, 0.1, 0.001], [0.1, 0.2, 0.9, 0.1]])

    # Non-square matrix without a value close to 1 in second column.
    A4 = torch.tensor([[0.1, 0.2, 0.9, 0.1], [1.1, 0.1, 0.1, 0.1], [0.1, 0.001, 0.1, 1.0]])
    expected4 = torch.tensor([[1.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.9, 0.1], [0.1, 0.001, 0.1, 1.0]])

    for A, expected in [
        (A0, expected0),
        (A1, expected1),
        (A2, expected2),
        (A3, expected3),
        (A4, expected4),
    ]:
        A_permuted = permute_to_identity(A)
        torch.testing.assert_close(A_permuted, expected)


def test_closness_to_identity():
    # Identity matrix
    A0 = torch.eye(3)
    assert pytest.approx(calculate_closeness_to_identity(A0)) == 0.0

    # 2x3 matrix which is close to the identity matrix
    A1 = torch.tensor([[0.95, 0.01, 0.0], [-0.02, 1.02, 0.01]])
    assert calculate_closeness_to_identity(A1) < 0.1

    # 3-tensor of 2x2 matrices close to the identity matrix
    A2 = torch.tensor([[[0.95, 0.01], [-0.02, 1.02]], [[0.99, 0.01], [-0.02, 1.01]]])
    # Should be equal to the mean of the closeness of each matrix separately
    full_closeness = calculate_closeness_to_identity(A2)
    closeness_0 = calculate_closeness_to_identity(A2[0])
    closeness_1 = calculate_closeness_to_identity(A2[1])
    assert pytest.approx(full_closeness) == (closeness_0 + closeness_1) / 2
