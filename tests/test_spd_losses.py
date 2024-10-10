import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from spd.run_spd import (
    calc_lp_sparsity_loss_rank_one,
    calc_orthog_loss_full_rank,
    calc_param_match_loss,
    calc_topk_l2_rank_one,
)


class TestCalcTopkL2:
    def test_calc_topk_l2_rank_one_single_instance_single_param_true_and_false(self):
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        topk_mask: Float[Tensor, "batch=1 k=2"] = torch.tensor(
            [[True, False, False]], dtype=torch.bool
        )
        n_params = 2 * 3 * 2
        result = calc_topk_l2_rank_one(
            As_and_Bs_vals=[(A, B)], topk_mask=topk_mask, n_params=n_params
        )

        # Below we write what the intermediate values are
        # A_topk = torch.tensor([[[1, 0, 0], [1, 0, 0]]])
        # AB_topk = torch.tensor([[[1, 1], [1, 1]]])
        # Sum and then divide by n_params: 4 / 12 = 1/3
        expected = torch.tensor(1.0 / 3.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_topk_l2_rank_one_single_instance_single_param_true_and_true(self):
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        topk_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
        n_params = 2 * 3 * 2
        result = calc_topk_l2_rank_one(
            As_and_Bs_vals=[(A, B)], topk_mask=topk_mask, n_params=n_params
        )

        # Below we write what the intermediate values are
        # A_topk = torch.tensor([[[1, 1, 1], [1, 1, 1]]])
        # AB_topk = torch.tensor([[[3, 3], [3, 3]]])
        # Square it, sum it and divide by n_params: 9*4 / 12 = 3
        expected = torch.tensor(3.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_topk_l2_rank_one_multiple_instances(self):
        A = torch.ones(2, 1, 2)
        B = torch.ones(2, 2, 1)
        # topk_mask: [batch=2, n_instances=2, k=2]
        topk_mask = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 1]]], dtype=torch.bool)
        n_params = 1 * 2 * 1
        result = calc_topk_l2_rank_one(
            As_and_Bs_vals=[(A, B)], topk_mask=topk_mask, n_params=n_params
        )

        # Below we write what the intermediate values are
        # A: [n_instances=2, d_in=1, k=2] = torch.tensor(
        #     [
        #         [[1, 1]],
        #         [[1, 1]]
        #     ]
        # )
        # A_topk: [batch=2, n_instances=2, d_in=1, k=2] = torch.tensor([
        #     [
        #         [[1, 0]],
        #         [[0, 1]]
        #     ],
        #     [
        #         [[0, 1]],
        #         [[1, 1]]
        #     ]
        # ])
        # B: [n_instances=2, k=2, d_out=1] = torch.tensor([
        #     [
        #         [[1]],
        #         [[1]]
        #     ],
        #     [
        #         [[1]],
        #         [[1]]
        #     ]
        # ])
        # AB_topk: [batch=2, n_instances=2, d_in=1, d_out=1] = torch.tensor([
        #     [
        #         [[1]],
        #         [[1]]
        #     ],
        #     [
        #         [[1]],
        #         [[2]]
        #     ]
        # ])
        # topk_l2_penalty (pre-reduction): [batch=2, n_instances=2] = torch.tensor([
        #     [1, 1],
        #     [1, 4]
        # ])
        # topk_l2_penalty (post-reduction): [n_instances=2] = torch.tensor([0.5, 1.25])
        expected = torch.tensor([0.5, 1.25])
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCalcParamMatchLoss:
    def test_calc_param_match_loss_single_instance_single_param(self):
        A = torch.ones(2, 3)
        B = torch.ones(3, 2)
        n_params = 2 * 3 * 2
        pretrained_weights = {"layer1": torch.tensor([[1.0, 1.0], [1.0, 1.0]])}
        subnetwork_params = {"layer1": A @ B}
        param_map = {"layer1": "layer1"}
        result = calc_param_match_loss(
            pretrained_weights=pretrained_weights,
            subnetwork_params_summed=subnetwork_params,
            param_map=param_map,
            has_instance_dim=False,
            n_params=n_params,
        )

        # A: [2, 3], B: [3, 2], both filled with ones
        # AB: [[3, 3], [3, 3]]
        # (AB - pretrained_weights)^2: [[4, 4], [4, 4]]
        # Sum and divide by n_params: 16 / 12 = 4/3
        expected = torch.tensor(4.0 / 3.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_single_instance_multiple_params(self):
        As = [torch.ones(2, 3), torch.ones(3, 3)]
        Bs = [torch.ones(3, 3), torch.ones(3, 2)]
        n_params = 2 * 3 * 3 + 3 * 3 * 2
        pretrained_weights = {
            "layer1": torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            "layer2": torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        }
        subnetwork_params = {
            "layer1": As[0] @ Bs[0],
            "layer2": As[1] @ Bs[1],
        }
        param_map = {"layer1": "layer1", "layer2": "layer2"}
        result = calc_param_match_loss(
            pretrained_weights=pretrained_weights,
            subnetwork_params_summed=subnetwork_params,
            param_map=param_map,
            has_instance_dim=False,
            n_params=n_params,
        )

        # First layer: AB1: [[3, 3, 3], [3, 3, 3]], diff^2: [[1, 1, 1], [1, 1, 1]]
        # Second layer: AB2: [[3, 3], [3, 3], [3, 3]], diff^2: [[4, 4], [4, 4], [4, 4]]
        # Add together 24 + 6 = 30
        # Divide by n_params: 30 / (18+18) = 5/6
        expected = torch.tensor(5.0 / 6.0)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

    def test_calc_param_match_loss_multiple_instances(self):
        As = [torch.ones(2, 2, 3)]
        Bs = [torch.ones(2, 3, 2)]
        n_params = 2 * 3 * 2
        pretrained_weights = {
            "layer1": torch.tensor([[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
        }
        subnetwork_params = {"layer1": As[0] @ Bs[0]}
        param_map = {"layer1": "layer1"}
        result = calc_param_match_loss(
            pretrained_weights=pretrained_weights,
            subnetwork_params_summed=subnetwork_params,
            param_map=param_map,
            has_instance_dim=True,
            n_params=n_params,
        )

        # AB [n_instances=2, d_in=2, d_out=2]: [[[3, 3], [3, 3]], [[3, 3], [3, 3]]]
        # diff^2: [[[1, 1], [1, 1]], [[4, 4], [4, 4]]]
        # Sum together and divide by n_params: [4, 16] / 12 = [1/3, 4/3]
        expected = torch.tensor([1.0 / 3.0, 4.0 / 3.0])
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


class TestCalcLpSparsityLoss:
    def test_calc_lp_sparsity_loss_rank_one_single_instance(self):
        inner_acts: dict[str, Float[Tensor, "batch=1 k=3"]] = {
            "layer": torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        }
        B_params: dict[str, Float[Tensor, "k=3 d_out=2"]] = {
            "layer": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        }

        # Compute layer_acts
        layer_acts = {"layer": inner_acts["layer"] @ B_params["layer"]}
        # Expected layer_acts: [9.0, 12.0]

        # Compute out (assuming identity activation function)
        out = layer_acts["layer"]
        # Expected out: [9.0, 12.0]

        step_pnorm = 0.9

        result = calc_lp_sparsity_loss_rank_one(
            out=out,
            layer_acts=layer_acts,
            inner_acts=inner_acts,
            B_params=B_params,
            step_pnorm=step_pnorm,
        )

        # Take derivative w.r.t each output dimension

        # d9/dlayer_acts = 1
        # d9/dinner_acts = [1, 3, 5]
        # attributions for 9 = [1, 3, 5] * [1, 1, 1] = [1, 3, 5]

        # d12/dinner_acts = [2, 4, 6]
        # attributions for 12 = [2, 4, 6] * [1, 1, 1] = [2, 4, 6]

        # Add the squared attributions = [1^2, 3^2, 5^2] + [2^2, 4^2, 6^2] = [5, 25, 61]
        # Divide by 2 (since we have two terms) = [2.5, 12.5, 30.5]
        # Take to the power of 0.9 and sqrt = [2.5^(0.9*0.5), 12.5^(0.9*0.5), 30.5^(0.9*0.5)]
        # Sum = 2.5^(0.9*0.5) + 12.5^(0.9*0.5) + 30.5^(0.9*0.5)

        expected_val = (2.5 ** (0.9 * 0.5)) + (12.5 ** (0.9 * 0.5)) + (30.5 ** (0.9 * 0.5))
        expected = torch.tensor(expected_val)
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

        # Also check that the B_params is in the computation graph of the sparsity result.
        result.backward()
        assert B_params["layer"].grad is not None


class TestCalcOrthogLoss:
    @pytest.mark.parametrize(
        "subnet, expected",
        [
            (torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), torch.tensor(0.0)),  # Orthogonal
            (torch.tensor([[[1.0, 0.0]], [[-4.0, 1.0]]]), torch.tensor(2.0)),  # Not orthogonal
        ],
    )
    def test_calc_orthog_loss_full_rank(self, subnet: Tensor, expected: Tensor) -> None:
        result = calc_orthog_loss_full_rank(subnet_param_vals=[subnet])
        # Note that we take the dot product, then abs, then mean over the subnetwork indices
        assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
