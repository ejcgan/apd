from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import nn

from spd.models.base import SPDModel
from spd.models.bool_circuit_models import MLPComponents
from spd.scripts.multilayer_functions.piecewise_linear import ControlledResNet


class PiecewiseSPDResNet(SPDModel):
    def __init__(
        self, n_inputs: int, d_mlp: int, num_layers: int, k: int, d_embed: int | None = None
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.num_layers = num_layers
        self.k = k
        self.d_embed = self.n_inputs + 1 if d_embed is None else d_embed
        self.d_control = self.d_embed - 2

        self.num_functions = n_inputs - 1
        self.n_outputs = 1  # this is hardcoded. This class isn't defined for multiple outputs

        self.superposition = self.num_functions > self.d_control
        if not self.superposition:
            assert self.num_functions == self.d_control

        self.W_E = nn.Linear(n_inputs, self.d_embed, bias=False)
        self.W_U = nn.Linear(self.d_embed, self.n_outputs, bias=False)

        self.initialise_embeds()

        self.layers = nn.ModuleList(
            [MLPComponents(self.d_embed, d_mlp, k) for _ in range(num_layers)]
        )

    def initialise_embeds(self):
        self.W_E.weight.data = torch.zeros(self.d_embed, self.n_inputs)
        self.W_E.weight.data[0, 0] = 1.0
        if not self.superposition:
            self.W_E.weight.data[1:-1, 1:] = torch.eye(self.num_functions)
        else:
            random_matrix = torch.randn(self.d_control, self.num_functions)
            random_normalised = random_matrix / torch.norm(random_matrix, dim=1, keepdim=True)
            self.W_E.weight.data[1:-1, 1:] = random_normalised

        self.W_U.weight.data = torch.zeros(self.n_outputs, self.d_embed)
        self.W_U.weight.data[:, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.W_E(x)
        for layer in self.layers:
            residual = residual + layer(residual)
        return self.W_U(residual)

    @property
    def all_decomposable_params(self) -> list[Float[torch.Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        params = []
        for mlp in self.layers:
            params.append(mlp.linear1.weight.T)
            params.append(mlp.linear2.weight.T)
        return params

    @classmethod
    def from_handcoded(cls, functions: list[Callable[[float], float]]) -> "PiecewiseSPDResNet":
        n_inputs = len(functions) + 2
        d_mlp = 32
        num_layers = 4
        k = 100
        d_embed = n_inputs
        start = 0
        end = 5
        model = cls(n_inputs=len(functions) + 2, d_mlp=32, num_layers=4, k=4)
        handcoded_model = ControlledResNet(
            functions,
            start=start,
            end=end,
            num_neurons=d_mlp,
            num_layers=num_layers,
            d_control=d_embed - 2,
        )
        # Copy the weights from the hand-coded model to the model
