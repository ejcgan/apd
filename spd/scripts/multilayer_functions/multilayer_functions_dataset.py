from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class PiecewiseDataset(Dataset[tuple[Float[Tensor, " n_inputs"], Float[Tensor, ""]]]):
    """Dataset for multilayer piecewise functions.

    The first bit is a real value and the rest of the bits are control bits indicating which of a
    set of boolean functions is on or off.
    """

    def __init__(
        self,
        n_inputs: int,
        functions: list[Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]],
        prob_one: float,
        range_max: int = 5,
        buffer_size: int = 1_000_000,
    ):
        self.n_inputs = n_inputs
        self.functions = functions
        self.prob_one = prob_one
        self.range_max = range_max
        self.buffer_size = buffer_size
        self.buffer = None
        self.buffer_index = 0

    def __len__(self) -> int:
        return 2**31

    def generate_buffer(self):
        data = torch.empty((self.buffer_size, self.n_inputs))
        data[:, 0] = torch.rand(self.buffer_size) * self.range_max
        control_bits = torch.bernoulli(
            torch.full((self.buffer_size, self.n_inputs - 1), self.prob_one)
        )
        data[:, 1:] = control_bits

        x = data[:, 0].unsqueeze(1).expand(-1, len(self.functions))
        function_outputs = torch.stack([f(x[:, i]) for i, f in enumerate(self.functions)], dim=1)

        labels = torch.einsum("bo,bo->b", control_bits, function_outputs)

        self.buffer = (data, labels)
        self.buffer_index = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.buffer is None or self.buffer_index >= self.buffer_size:
            self.generate_buffer()

        assert self.buffer is not None
        item = (self.buffer[0][self.buffer_index], self.buffer[1][self.buffer_index])
        self.buffer_index += 1
        return item
