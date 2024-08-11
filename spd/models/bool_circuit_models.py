import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.log import logger
from spd.models.base import Model, SPDModel
from spd.models.components import MLPComponents
from spd.scripts.bool_circuits.bool_circuit_utils import BooleanOperation, make_detailed_circuit


class MLP(nn.Module):
    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_mlp)
        self.linear2 = nn.Linear(d_mlp, d_embed, bias=False)  # No bias in down-projection

    def forward(self, x: Float[Tensor, "... d_embed"]) -> Float[Tensor, "... d_embed"]:
        return self.linear2(F.relu(self.linear1(x)))


class BoolCircuitTransformer(Model):
    def __init__(self, n_inputs: int, d_embed: int, d_mlp: int, n_layers: int, n_outputs: int = 1):
        super().__init__()
        self.n_inputs = n_inputs
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        self.W_E = nn.Linear(n_inputs, d_embed, bias=False)
        self.W_U = nn.Linear(d_embed, n_outputs, bias=False)
        self.layers = nn.ModuleList([MLP(d_embed, d_mlp) for _ in range(n_layers)])

    def forward(self, x: Float[Tensor, "batch inputs"]) -> Float[Tensor, "batch outputs"]:
        residual = self.W_E(x)
        for layer in self.layers:
            residual = residual + layer(residual)
        return self.W_U(residual)

    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        """List of all parameters which will be decomposed with SPD."""
        params = []
        for mlp in self.layers:
            params.append(mlp.linear1.weight.T)
            params.append(mlp.linear2.weight.T)
        return params

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BoolCircuitTransformer":
        path = Path(path)
        with open(path.parent / "config.json") as f:
            config = json.load(f)

        params = torch.load(path, map_location="cpu")

        model = cls(
            n_inputs=config["n_inputs"],
            d_embed=config["d_embed"],
            d_mlp=config["d_mlp"],
            n_layers=config["n_layers"],
            n_outputs=config["n_outputs"],
        )
        model.load_state_dict(params)
        return model

    def init_handcoded(self, circuit: list[BooleanOperation]) -> None:
        detailed_circuit: list[BooleanOperation] = make_detailed_circuit(circuit, self.n_inputs)

        device = self.W_E.weight.device

        assert len(detailed_circuit) + self.n_inputs <= self.d_embed, "d_embed is too low"

        self.W_E.weight.data[: self.n_inputs, :] = torch.eye(self.n_inputs, device=device)
        self.W_E.weight.data[self.n_inputs :, :] = torch.zeros(
            self.d_embed - self.n_inputs, self.n_inputs, device=device
        )

        assert self.n_outputs == 1, "Only one output supported"
        out_idx = detailed_circuit[-1].out_idx
        self.W_U.weight.data = torch.zeros(self.n_outputs, self.d_embed, device=device)
        self.W_U.weight.data[0, out_idx] = 1.0

        for op in detailed_circuit:
            if op.min_layer_needed is None:
                raise ValueError("min_layer_needed not set")
            assert op.min_layer_needed < self.n_layers, "Not enough layers"
        for i in range(self.n_layers):
            # torch.nn shapes are (d_output, d_input)
            # linear1.shape = (d_mlp, d_embed)
            # linear2.shape = (d_embed, d_mlp)
            self.layers[i].linear1.weight.data = torch.zeros(
                self.d_mlp, self.d_embed, device=device
            )
            self.layers[i].linear1.bias.data = torch.zeros(self.d_mlp, device=device)
            self.layers[i].linear2.weight.data = torch.zeros(
                self.d_embed, self.d_mlp, device=device
            )

        used_neurons = [0 for _ in range(self.n_layers)]
        for op in detailed_circuit:
            gate = op.name
            arg1 = op.input_idx1
            arg2 = op.input_idx2
            out_idx = op.out_idx
            min_layer = op.min_layer_needed
            assert min_layer is not None, "min_layer_needed not set"
            if gate == "AND":
                assert arg2 is not None, "AND gate requires two arguments"
                neuron_index = used_neurons[min_layer]
                used_neurons[min_layer] += 1
                self.layers[min_layer].linear1.weight.data[neuron_index, [arg1, arg2]] = 1.0
                self.layers[min_layer].linear1.bias.data[neuron_index] = -1.0
                self.layers[min_layer].linear2.weight.data[out_idx, neuron_index] = 1.0
            elif gate == "NOT":
                neuron_index = used_neurons[min_layer]
                used_neurons[min_layer] += 1
                self.layers[min_layer].linear1.weight.data[neuron_index, arg1] = -1.0
                self.layers[min_layer].linear1.bias.data[neuron_index] = 1.0
                self.layers[min_layer].linear2.weight.data[out_idx, neuron_index] = 1.0
            elif gate == "OR":
                assert arg2 is not None, "OR gate requires two arguments"
                neuron_index_ANDOR = used_neurons[min_layer]
                used_neurons[min_layer] += 1
                neuron_index_AND = used_neurons[min_layer]
                used_neurons[min_layer] += 2
                self.layers[min_layer].linear1.weight.data[neuron_index_ANDOR, [arg1, arg2]] = 1.0
                self.layers[min_layer].linear1.bias.data[neuron_index_ANDOR] = 0
                self.layers[min_layer].linear1.weight.data[neuron_index_AND, [arg1, arg2]] = 1.0
                self.layers[min_layer].linear1.bias.data[neuron_index_AND] = -1
                self.layers[min_layer].linear2.weight.data[out_idx, neuron_index_ANDOR] = 1.0
                self.layers[min_layer].linear2.weight.data[out_idx, neuron_index_AND] = -1.0
            else:
                raise ValueError(f"Unknown gate {gate}")
        logger.info(f"Used neurons per layer: {used_neurons}")


class BoolCircuitSPDTransformer(SPDModel):
    def __init__(
        self, n_inputs: int, d_embed: int, d_mlp: int, n_layers: int, k: int, n_outputs: int = 1
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_param_matrices = n_layers * 2
        self.k = k
        self.n_outputs = n_outputs

        self.W_E = nn.Linear(n_inputs, d_embed, bias=False)
        self.W_U = nn.Linear(d_embed, n_outputs, bias=False)
        self.layers = nn.ModuleList([MLPComponents(d_embed, d_mlp, k) for _ in range(n_layers)])

    def all_As(self) -> list[Float[Tensor, "dim k"]]:
        all_A_pairs = [
            (self.layers[i].linear1.A, self.layers[i].linear2.A) for i in range(self.n_layers)
        ]
        As = [A for A_pair in all_A_pairs for A in A_pair]
        assert len(As) == self.n_param_matrices
        return As

    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        all_B_pairs = [
            (self.layers[i].linear1.B, self.layers[i].linear2.B) for i in range(self.n_layers)
        ]
        As = [B for B_pair in all_B_pairs for B in B_pair]
        assert len(As) == self.n_param_matrices
        return As

    def forward(
        self, x: Float[Tensor, "... inputs"]
    ) -> tuple[
        Float[Tensor, "... outputs"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Returns:
            x: The output of the MLP
            layer_acts: A list of activations for each layer in each MLP.
            inner_acts: A list of component activations for each layer in each MLP.
        """
        layer_acts = []
        inner_acts = []
        residual = self.W_E(x)
        for layer in self.layers:
            layer_out, layer_acts_i, inner_acts_i = layer(residual)
            residual = residual + layer_out
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)
        return self.W_U(residual), layer_acts, inner_acts

    def forward_topk(
        self, x: Float[Tensor, "... inputs"], topk_indices: Int[Tensor, "... topk"]
    ) -> tuple[
        Float[Tensor, "... outputs"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each component activation.

        Args:
            x: Input tensor
            topk_indices: Boolean tensor indicating which components to keep

        Returns:
            output: The output of the transformer
            layer_acts: A list of activations for each layer in each MLP
            inner_acts: A list of component activations for each layer in each MLP
        """
        layer_acts = []
        inner_acts = []
        residual = self.W_E(x)

        for i, layer in enumerate(self.layers):
            layer_out, layer_acts_i, inner_acts_i = layer.forward_topk(residual, topk_indices)
            residual = residual + layer_out
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)

        return self.W_U(residual), layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BoolCircuitSPDTransformer":
        path = Path(path)
        with open(path.parent / "config.json") as f:
            config = json.load(f)

        params = torch.load(path, weights_only=True, map_location="cpu")

        model = cls(
            n_inputs=config["n_inputs"],
            d_embed=config["d_embed"],
            d_mlp=config["d_mlp"],
            n_layers=config["n_layers"],
            k=config["k"],
            n_outputs=config["n_outputs"],
        )
        model.load_state_dict(params)
        return model
