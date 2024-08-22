from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.run_spd import SPDModel, calc_topk_l2


class DummySPDModel(SPDModel):
    def __init__(self, d_in: int, d_out: int, k: int, n_instances: int | None = None):
        self.n_instances = n_instances
        self.n_param_matrices = 1
        self.d_in = d_in
        self.d_out = d_out
        self.k = k

    def all_As(self) -> list[Float[torch.Tensor, "... d_in k"]]:
        if self.n_instances is None:
            return [torch.ones(self.d_in, self.k)]
        return [torch.ones(self.n_instances, self.d_in, self.k)]

    def all_Bs(self) -> list[Float[torch.Tensor, "... k d_out"]]:
        if self.n_instances is None:
            return [torch.ones(self.k, self.d_out)]
        return [torch.ones(self.n_instances, self.k, self.d_out)]

    def forward_topk(
        self, x: Float[Tensor, "... dim"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... dim"],
        list[Float[Tensor, "... dim"]],
        list[Float[Tensor, "... k"]],
    ]:
        return x, [x], [topk_mask]

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DummySPDModel":
        return cls(d_in=1, d_out=1, k=1)


def test_calc_topk_l2_single_instance_single_param_true_and_false():
    model = DummySPDModel(d_in=2, d_out=2, k=3)
    topk_mask: Float[Tensor, "batch=1 k=2"] = torch.tensor([[True, False, False]], dtype=torch.bool)
    result = calc_topk_l2(model, topk_mask, device="cpu")

    # Below we write what the intermediate values are
    # A_topk = torch.tensor([[[1, 0, 0], [1, 0, 0]]])
    # AB_topk = torch.tensor([[[1, 1], [1, 1]]])
    expected = torch.tensor(1.0)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_calc_topk_l2_single_instance_single_param_true_and_true():
    model = DummySPDModel(d_in=2, d_out=2, k=3)
    topk_mask = torch.tensor([[True, True, True]], dtype=torch.bool)
    result = calc_topk_l2(model, topk_mask, device="cpu")

    # Below we write what the intermediate values are
    # A_topk = torch.tensor([[[1, 1, 1], [1, 1, 1]]])
    # AB_topk = torch.tensor([[[3, 3], [3, 3]]])
    expected = torch.tensor(9.0)
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"


def test_calc_topk_l2_multiple_instances():
    model = DummySPDModel(d_in=1, d_out=1, n_instances=2, k=2)
    # topk_mask: [batch=2, n_instances=2, k=2]
    topk_mask = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 1]]], dtype=torch.bool)
    result = calc_topk_l2(model, topk_mask, device="cpu")

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
    # topk_l2_penalty (post-reduction): [n_instances=2] = torch.tensor([1, 2.5])
    expected = torch.tensor([1.0, 2.5])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"
