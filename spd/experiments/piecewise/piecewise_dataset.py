from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class PiecewiseDataset(Dataset[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]]):
    """Dataset for multilayer piecewise functions.

    The first bit is a real value and the rest of the bits are control bits indicating which of a
    set of boolean functions is on or off.

    Uses a buffer to save on expensive calls to torch.randn and torch.bernoulli.
    """

    def __init__(
        self,
        n_inputs: int,
        functions: list[Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]],
        feature_probability: float,
        range_min: float,
        range_max: float,
        batch_size: int,
        return_labels: bool,
    ):
        self.n_inputs = n_inputs
        self.functions = functions
        self.feature_probability = feature_probability
        self.range_min = range_min
        self.range_max = range_max
        self.batch_size = batch_size
        self.return_labels = return_labels

    def __len__(self) -> int:
        return 2**31

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = torch.empty((self.batch_size, self.n_inputs))
        data[:, 0] = (
            torch.rand(self.batch_size) * (self.range_max - self.range_min) + self.range_min
        )
        control_bits = torch.bernoulli(
            torch.full((self.batch_size, self.n_inputs - 1), self.feature_probability)
        )
        data[:, 1:] = control_bits
        if self.return_labels:
            x = data[:, 0].unsqueeze(1).expand(-1, len(self.functions))
            function_outputs = torch.stack(
                [f(x[:, i]) for i, f in enumerate(self.functions)], dim=-1
            )

            labels = torch.einsum("bo,bo->b", control_bits, function_outputs).unsqueeze(1)
        else:
            labels = torch.empty(0)
        return data, labels
