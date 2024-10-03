from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.base import SPDFullRankModel
from spd.utils import (
    calc_ablation_attributions,
    calc_attributions_rank_one,
    calc_topk_mask,
    calculate_closeness_to_identity,
    permute_to_identity,
)


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


def test_calc_attributions_rank_one_one_inner_act():
    # Set up a simple linear model with known gradients
    inner_acts = [torch.tensor([2.0, 3.0], requires_grad=True)]

    # Define weights for our linear model
    weights = torch.tensor(
        [
            [1.0, 2.0],  # For out[0]
            [3.0, 4.0],  # For out[1]
        ]
    )

    # Calculate the output
    out = torch.matmul(weights, inner_acts[0])

    # Calculate attributions
    attributions = calc_attributions_rank_one(out, inner_acts)

    # Expected attributions
    expected_attributions = torch.zeros_like(inner_acts[0])
    for i in range(2):  # For each output dimension
        attribution_per_dim = weights[i] * inner_acts[0]
        expected_attributions += attribution_per_dim**2

    # Check if the calculated attributions match the expected attributions
    torch.testing.assert_close(attributions, expected_attributions)

    # Additional check: ensure the shape is correct
    assert attributions.shape == inner_acts[0].shape


def test_calc_attributions_rank_one_two_inner_acts():
    # Set up a simple linear model with known gradients
    inner_acts = [
        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
    ]

    # Define weights for our linear model
    weights = [
        torch.tensor([2.0, 3.0]),  # Gradients will be 2 and 3 for the first inner_act
        torch.tensor([1.0, 4.0]),  # Gradients will be 1 and 4 for the second inner_act
    ]

    # Calculate the output
    out = torch.stack(
        [
            inner_acts[0] * weights[0][0] + inner_acts[1] * weights[1][0],
            inner_acts[0] * weights[0][1] + inner_acts[1] * weights[1][1],
        ],
        dim=-1,
    )

    # Calculate attributions
    attributions = calc_attributions_rank_one(out, inner_acts)

    # Expected attributions
    expected_attributions = torch.zeros_like(inner_acts[0])
    for i in range(2):  # For each output dimension
        attribution_per_dim = sum(weights[j][i] * inner_acts[j] for j in range(2))
        expected_attributions += attribution_per_dim**2

    # Check if the calculated attributions match the expected attributions
    torch.testing.assert_close(attributions, expected_attributions)

    # Additional check: ensure the shape is correct
    assert attributions.shape == inner_acts[0].shape


def test_calc_topk_mask_without_batch_topk():
    attribution_scores = torch.tensor([[1.0, 5.0, 2.0, 1.0, 2.0], [3.0, 3.0, 5.0, 4.0, 4.0]])
    topk = 3
    expected_mask = torch.tensor(
        [[False, True, True, False, True], [False, False, True, True, True]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=False)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_with_batch_topk():
    attribution_scores = torch.tensor([[1.0, 5.0, 2.0, 1.0, 2.0], [3.0, 3.0, 5.0, 4.0, 4.0]])
    topk = 3  # mutliplied by batch size to get 6
    expected_mask = torch.tensor(
        [[False, True, False, False, False], [True, True, True, True, True]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=True)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_without_batch_topk_n_instances():
    """attributions have shape [batch, n_instances, n_features]. We take the topk
    over the n_features dim for each instance in each batch."""
    attribution_scores = torch.tensor(
        [[[1.0, 5.0, 3.0, 4.0], [2.0, 4.0, 6.0, 1.0]], [[2.0, 1.0, 5.0, 9.5], [3.0, 4.0, 1.0, 5.0]]]
    )
    topk = 2
    expected_mask = torch.tensor(
        [
            [[False, True, False, True], [False, True, True, False]],
            [[False, False, True, True], [False, True, False, True]],
        ]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=False)
    torch.testing.assert_close(result, expected_mask)


def test_calc_topk_mask_with_batch_topk_n_instances():
    """attributions have shape [batch, n_instances, n_features]. We take the topk
    over the concatenated batch and n_features dim."""
    attribution_scores = torch.tensor(
        [[[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]], [[2.0, 1.0, 5.0], [3.0, 4.0, 1.0]]]
    )
    topk = 2  # multiplied by batch size to get 4
    expected_mask = torch.tensor(
        [[[False, True, True], [False, True, True]], [[True, False, True], [True, True, False]]]
    )

    result = calc_topk_mask(attribution_scores, topk, batch_topk=True)
    torch.testing.assert_close(result, expected_mask)


def test_ablation_attributions():
    class TestModel(SPDFullRankModel):
        def __init__(self):
            super().__init__()
            self.subnetwork_params: Float[Tensor, "n_subnets dim"] = nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            )
            self.k = 2

        def forward(self, x):  # type: ignore
            out = torch.einsum("i,ki->", x, self.subnetwork_params)
            return out, None, None

        def forward_topk(self, x, topk_mask):  # type: ignore
            raise NotImplementedError

        @classmethod
        def from_pretrained(cls, path: str | Path) -> "SPDFullRankModel":
            raise NotImplementedError

        def all_subnetwork_params(self) -> dict[str, Float[Tensor, "... k d_layer_in d_layer_out"]]:
            raise NotImplementedError

        def set_subnet_to_zero(self, subnet_idx: int) -> dict[str, Tensor]:
            stored_vals = {"subnetwork_params": self.subnetwork_params[subnet_idx].detach().clone()}
            self.subnetwork_params[subnet_idx] = 0.0
            return stored_vals

        def restore_subnet(self, subnet_idx: int, stored_vals: dict[str, Tensor]) -> None:
            self.subnetwork_params[subnet_idx] = stored_vals["subnetwork_params"]

    model = TestModel()
    batch = torch.tensor([1.0, 1.0])
    out, _, _ = model(batch)
    attributions = calc_ablation_attributions(model, batch, out)

    # output with all subnets:
    # [1.0, 1.0] @ [1.0, 2.0] + [1.0, 1.0] @ [3.0, 4.0] = [10]
    # output without subnet0: [1.0, 1.0] @ [3.0, 4.0] = [7.0]
    # output without subnet1: [1.0, 1.0] @ [1.0, 2.0] = [3.0]
    # attributions:
    # 0. (10 - 7) ** 2 = 9
    # 1. (10 - 3) ** 2 = 49
    expected_attributions = torch.tensor([9.0, 49.0])
    torch.testing.assert_close(attributions, expected_attributions)
