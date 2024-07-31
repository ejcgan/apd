import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.log import logger
from spd.scripts.bool_circuits.bool_circuit_utils import BooleanOperation, make_detailed_circuit


class MLP(nn.Module):
    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_mlp)
        self.linear2 = nn.Linear(d_mlp, d_embed, bias=False)  # No bias in down-projection

    def forward(self, x: Float[Tensor, "... d_embed"]) -> Float[Tensor, "... d_embed"]:
        return self.linear2(F.relu(self.linear1(x)))


class BoolCircuitTransformer(nn.Module):
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
