from collections.abc import Callable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

# from spd.scripts.multilayer_functions.spd_training import PiecewiseFunctionTransformer
from spd.models.piecewise_models import PiecewiseFunctionTransformer

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


def generate_trig_functions(
    num_trig_functions: int,
) -> list[Callable[[float | Float[Tensor, ""]], float]]:
    def create_trig_function(
        a: float, b: float, c: float, d: float, e: float, f: float, g: float
    ) -> Callable[[float], float]:
        return lambda x: a * np.sin(b * x + c) + d * np.cos(e * x + f) + g

    trig_functions = []
    for _ in range(num_trig_functions):
        a = np.random.uniform(-1, 1)
        b = np.exp(np.random.uniform(-1, 3))
        c = np.random.uniform(-np.pi, np.pi)
        d = np.random.uniform(-1, 1)
        e = np.exp(np.random.uniform(-1, 3))
        f = np.random.uniform(-np.pi, np.pi)
        g = np.random.uniform(-1, 1)
        trig_functions.append(create_trig_function(a, b, c, d, e, f, g))
    return trig_functions


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


# %% Do it with the new class

num_functions = 3
trigs = generate_trig_functions(num_functions)
test = PiecewiseFunctionTransformer.from_handcoded(trigs)

control_bits = torch.ones(num_functions, dtype=torch.float32)
# test.plot(-0.1, 5.1, 1000, control_bits=control_bits, functions=trigs)
test.plot_multiple(-0.1, 5.1, 200, control_bits=control_bits, functions=trigs)

# %%
from torch.utils.data import DataLoader

from spd.scripts.multilayer_functions.multilayer_functions_decomposition import PiecewiseDataset

functions = trigs
ds = PiecewiseDataset(4, functions, prob_one=0.5)
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
