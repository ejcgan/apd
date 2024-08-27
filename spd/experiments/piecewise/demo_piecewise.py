from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from spd.experiments.piecewise.models import PiecewiseFunctionTransformer
from spd.experiments.piecewise.piecewise_decomposition import (
    PiecewiseDataset,
    generate_trig_functions,
)


# %%
# test
# make a list of 50 different cubic functions
def generate_cubics(num_cubics: int) -> list[Callable[[float], float]]:
    def create_cubic(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
        return lambda x: a * x**3 + b * x**2 + c * x + d

    cubics = []
    for _ in range(num_cubics):
        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-2, 2)
        c = np.random.uniform(-4, 4)
        d = np.random.uniform(-8, 8)
        cubics.append(create_cubic(a, b, c, d))
    return cubics


def generate_regular_simplex(num_vertices: int) -> torch.Tensor:
    # Create the standard basis in num_vertices dimensions
    basis = torch.eye(num_vertices)

    # Create the (1,1,...,1) vector
    ones = torch.ones(num_vertices)

    # Compute the Householder transformation
    v = ones / torch.norm(ones)
    last_basis_vector = torch.zeros(num_vertices)
    last_basis_vector[-1] = 1
    u = v - last_basis_vector
    u = u / torch.norm(u)

    # Apply the Householder transformation
    H = torch.eye(num_vertices) - 2 * u.outer(u)
    rotated_basis = basis @ H

    # Remove the last coordinate
    simplex = rotated_basis[:, :-1]

    # Center the simplex at the origin
    centroid = simplex.mean(dim=0)
    simplex = simplex - centroid

    return simplex / simplex.norm(dim=1).unsqueeze(1)


# %%
num_functions = 5
neurons_per_function = 100
feature_probability = 0.2
functions, _ = generate_trig_functions(num_functions)
test = PiecewiseFunctionTransformer.from_handcoded(
    functions=functions,
    neurons_per_function=neurons_per_function,
    n_layers=2,
    range_min=0,
    range_max=5,
)

# test.plot(-0.1, 5.1, 1000, control_bits=control_bits, functions=functions)
test.plot_multiple(start=-0, end=5.1, num_points=200, functions=functions, prob=feature_probability)

# %%
ds = PiecewiseDataset(4, functions, feature_probability=0.5, range_min=0, range_max=5)
dl = DataLoader(ds, batch_size=1, shuffle=False)
for i, (batch, labels) in enumerate(dl):
    print("Batch")
    print(batch)
    print("labels")
    print(labels)
    # Check that the model outputs the same
    model_out = test(batch)
    print("model_out")
    print(model_out)
    print("\n")

    if i > 10:
        break
# %%
