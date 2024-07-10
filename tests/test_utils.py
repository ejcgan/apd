import pytest
import torch

from spd.utils import calculate_closeness_to_identity, permute_to_identity


@pytest.mark.parametrize(
    "A, expected",
    [
        (torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]), torch.eye(3)),
        (torch.tensor([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]]), torch.eye(3)),
        (
            torch.tensor([[0, 0.9, 0.0], [0.9, 1, 0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0.9, 1, 0.0], [0, 0.9, 0.0], [0.0, 0.0, 1.0]]),
        ),
        (
            torch.tensor(
                [
                    [0.1, 0.2, 0.9, 0.1],
                    [0.8, 0.1, 0.1, 0.1],
                    [0.1, 0.1, 0.1, 0.9],
                    [0.1, 0.9, 0.1, 0.1],
                ]
            ),
            torch.tensor(
                [
                    [0.8, 0.1, 0.1, 0.1],
                    [0.1, 0.9, 0.1, 0.1],
                    [0.1, 0.2, 0.9, 0.1],
                    [0.1, 0.1, 0.1, 0.9],
                ]
            ),
        ),
    ],
)
def test_permute_to_identity(A: torch.Tensor, expected: torch.Tensor):
    A_permuted = permute_to_identity(A)
    torch.testing.assert_close(A_permuted, expected)


@pytest.mark.parametrize(
    "A, expected_closeness_max",
    [
        (torch.eye(3), 0.0),
        (torch.tensor([[0.95, 0.01, 0.0], [-0.02, 1.02, 0.01], [0.0, 0.0, 1.0]]), 0.1),
        (torch.tensor([[1.0, 0.0], [1.0, 0.0]]), 2 ** (1 / 2)),
    ],
)
def test_closeness_to_identity(A: torch.Tensor, expected_closeness_max: float):
    closeness = calculate_closeness_to_identity(A)
    assert closeness <= expected_closeness_max
