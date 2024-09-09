from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor

from spd.types import TrigParams


def create_trig_function(
    a: float, b: float, c: float, d: float, e: float, f: float, g: float
) -> Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]:
    return lambda x: a * torch.sin(b * x + c) + d * torch.cos(e * x + f) + g


def generate_trig_functions(
    num_trig_functions: int,
) -> tuple[
    list[Callable[[Float[Tensor, " n_inputs"]], Float[Tensor, " n_inputs"]]],
    list[TrigParams],
]:
    trig_functions = []
    params = []
    for _ in range(num_trig_functions):
        a = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        b = torch.rand(1).item() * 0.9 + 0.1  # Uniform(0.1, 1)
        c = torch.rand(1).item() * 2 * torch.pi - torch.pi  # Uniform(-π, π)
        d = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        e = torch.rand(1).item() * 0.9 + 0.1  # Uniform(0.1, 1)
        f = torch.rand(1).item() * 2 * torch.pi - torch.pi  # Uniform(-π, π)
        g = torch.rand(1).item() * 2 - 1  # Uniform(-1, 1)
        trig_functions.append(create_trig_function(a, b, c, d, e, f, g))
        params.append((a, b, c, d, e, f, g))
    return trig_functions, params
