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
        buffer_size: int = 1_000_000,
    ):
        self.n_inputs = n_inputs
        self.functions = functions
        self.feature_probability = feature_probability
        self.range_min = range_min
        self.range_max = range_max
        self.buffer_size = buffer_size
        self.buffer = None
        self.buffer_index = 0

    def __len__(self) -> int:
        return 2**31

    def generate_buffer(self):
        data = torch.empty((self.buffer_size, self.n_inputs))
        data[:, 0] = (
            torch.rand(self.buffer_size) * (self.range_max - self.range_min) + self.range_min
        )
        original_control_bits = torch.empty((self.buffer_size, self.n_inputs - 1))
        original_control_bits.bernoulli_(self.feature_probability)
        # Ensure at least one control bit is on by removing all the rows with all zeros, then
        # generating new control bits for those rows and repeating until all rows have at least one
        # bit on.
        i = 0
        target_length = original_control_bits.shape[0]
        current_control_bits = original_control_bits[original_control_bits.any(dim=1)]
        while current_control_bits.shape[0] < target_length:
            new_control_bits = torch.empty((self.buffer_size, self.n_inputs - 1))
            new_control_bits.bernoulli_(self.feature_probability)
            new_nonzero_control_bits = new_control_bits[new_control_bits.any(dim=1)]
            current_control_bits = torch.cat(
                (current_control_bits, new_nonzero_control_bits), dim=0
            )
            i += 1
        control_bits = current_control_bits[:target_length]
        data[:, 1:] = control_bits

        x = data[:, 0].unsqueeze(1).expand(-1, len(self.functions))
        function_outputs = torch.stack([f(x[:, i]) for i, f in enumerate(self.functions)], dim=-1)

        labels = torch.einsum("bo,bo->b", control_bits, function_outputs).unsqueeze(1)

        self.buffer = (data, labels)
        self.buffer_index = 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.buffer is None or self.buffer_index >= self.buffer_size:
            self.generate_buffer()

        assert self.buffer is not None
        item = (self.buffer[0][self.buffer_index], self.buffer[1][self.buffer_index])
        self.buffer_index += 1
        return item


def test_removing_zeros():
    n_inputs = 5
    functions = [lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4]
    feature_probability = 0.1
    range_min = -1
    range_max = 1
    buffer_size = 1000

    dataset = PiecewiseDataset(
        n_inputs, functions, feature_probability, range_min, range_max, buffer_size
    )

    for i in range(10):
        x, y = dataset[i]
        print(x, y)


if __name__ == "__main__":
    test_removing_zeros()
