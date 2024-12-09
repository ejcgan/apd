from pathlib import Path

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.base import SPDFullRankModel
from spd.utils import (
    SparseFeatureDataset,
    calc_ablation_attributions,
    calc_activation_attributions,
    calc_topk_mask,
    calculate_closeness_to_identity,
    compute_feature_importances,
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

        def forward(self, x, topk_mask=None):  # type: ignore
            out = torch.einsum("i,ki->", x, self.subnetwork_params)
            return out, None, None

        @classmethod
        def from_pretrained(cls, path: str | Path) -> "SPDFullRankModel":
            raise NotImplementedError

        def all_subnetwork_params(self) -> dict[str, Float[Tensor, "... k d_layer_in d_layer_out"]]:
            raise NotImplementedError

        def all_subnetwork_params_summed(
            self,
        ) -> dict[
            str,
            Float[Tensor, "d_layer_in d_layer_out"]
            | Float[Tensor, "n_instances d_layer_in d_layer_out"],
        ]:
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


def test_calc_activation_attributions_obvious():
    inner_acts = {"layer1": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])}
    expected = torch.tensor([[1.0, 1.0]])

    result = calc_activation_attributions(inner_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_different_d_out():
    inner_acts = {
        "layer1": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        "layer2": torch.tensor([[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]]),
    }
    expected = torch.tensor(
        [[1.0**2 + 2**2 + 5**2 + 6**2 + 7**2, 3**2 + 4**2 + 8**2 + 9**2 + 10**2]]
    )

    result = calc_activation_attributions(inner_acts)
    torch.testing.assert_close(result, expected)


def test_calc_activation_attributions_with_n_instances():
    # Batch=1, n_instances=2, k=2, d_out=2
    inner_acts = {
        "layer1": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]),
        "layer2": torch.tensor([[[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]]),
    }
    expected = torch.tensor(
        [
            [
                [1.0**2 + 2**2 + 9**2 + 10**2, 3**2 + 4**2 + 11**2 + 12**2],
                [5**2 + 6**2 + 13**2 + 14**2, 7**2 + 8**2 + 15**2 + 16**2],
            ]
        ]
    )

    result = calc_activation_attributions(inner_acts)
    torch.testing.assert_close(result, expected)


def test_dataset_at_least_zero_active():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5
    device = "cpu"
    batch_size = 100

    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
        value_range=(0.0, 1.0),
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that the proportion of non-zero elements is close to feature_probability
    non_zero_proportion = torch.count_nonzero(batch) / batch.numel()
    assert (
        abs(non_zero_proportion - feature_probability) < 0.05
    ), f"Expected proportion {feature_probability}, but got {non_zero_proportion}"


def test_dataset_exactly_one_active():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5  # This won't be used when data_generation_type="exactly_one_active"
    device = "cpu"
    batch_size = 10
    value_range = (-1.0, 3.0)

    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="exactly_one_active",
        value_range=value_range,
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that there's exactly one non-zero value per sample and instance
    for sample in batch:
        for instance in sample:
            non_zero_count = torch.count_nonzero(instance)
            assert non_zero_count == 1, f"Expected 1 non-zero value, but found {non_zero_count}"

    # Check that the non-zero values are in the value_range
    non_zero_values = batch[batch != 0]
    assert torch.all(
        (non_zero_values >= value_range[0]) & (non_zero_values <= value_range[1])
    ), f"Non-zero values should be between {value_range[0]} and {value_range[1]}"


def test_dataset_exactly_two_active():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5  # This won't be used when data_generation_type="exactly_one_active"
    device = "cpu"
    batch_size = 10
    value_range = (0.0, 1.0)

    dataset = SparseFeatureDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="exactly_two_active",
        value_range=value_range,
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that there's exactly one non-zero value per sample and instance
    for sample in batch:
        for instance in sample:
            non_zero_count = torch.count_nonzero(instance)
            assert non_zero_count == 2, f"Expected 2 non-zero values, but found {non_zero_count}"

    # Check that the non-zero values are in the value_range
    non_zero_values = batch[batch != 0]
    assert torch.all(
        (non_zero_values >= value_range[0]) & (non_zero_values <= value_range[1])
    ), f"Non-zero values should be between {value_range[0]} and {value_range[1]}"


@pytest.mark.parametrize(
    "importance_val, expected_tensor",
    [
        (
            1.0,
            torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        ),
        (
            0.5,
            torch.tensor(
                [[[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]], [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]]
            ),
        ),
        (
            0.0,
            torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        ),
    ],
)
def test_compute_feature_importances(
    importance_val: float, expected_tensor: Float[Tensor, "batch_size n_instances n_features"]
):
    importances = compute_feature_importances(
        batch_size=2, n_instances=2, n_features=3, importance_val=importance_val, device="cpu"
    )
    torch.testing.assert_close(importances, expected_tensor)
