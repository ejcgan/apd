import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

# %%
from spd.scripts.bool_circuits.circuit_utils import make_detailled_circuit


class MLP(nn.Module):
    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_mlp)  # W.shape = (d_mlp, d_embed)
        self.linear2 = nn.Linear(d_mlp, d_embed, bias=False)  # W.shape = (d_embed, d_mlp)

    def forward(self, x: Float[Tensor, "... d_embed"]) -> Float[Tensor, "... d_embed"]:
        return self.linear2(F.relu(self.linear1(x)))


class Transformer(nn.Module):
    def __init__(self, n_inputs: int, d_embed: int, d_mlp: int, n_layers: int, n_outputs: int = 1):
        super().__init__()
        self.n_inputs = n_inputs
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_outputs = n_outputs
        self.n_layers = n_layers

        self.W_E = nn.Linear(n_inputs, d_embed, bias=False)  # W_E.shape = (d_embed, n_inputs)
        self.W_U = nn.Linear(d_embed, n_outputs, bias=False)  # W_U.shape = (n_outputs, d_embed)
        self.layers = nn.ModuleList([MLP(d_embed, d_mlp) for _ in range(n_layers)])

    def forward(self, x: Float[Tensor, "batch inputs"]) -> Float[Tensor, "batch outputs"]:
        residual = self.W_E(x)
        for layer in self.layers:
            residual = residual + layer(residual)
        return self.W_U(residual)


class BoolCircuitModel(Transformer):
    def __init__(self, n_inputs: int, d_hidden: int, n_layers: int):
        super().__init__(
            n_inputs=n_inputs, d_embed=d_hidden, d_mlp=d_hidden, n_layers=n_layers, n_outputs=1
        )

    def hand_coded_implementation(self, circuit):
        detailled_circuit = make_detailled_circuit(circuit, self.n_inputs)

        assert len(detailled_circuit) + self.n_inputs <= self.d_embed, "d_hidden is too low"

        print(f"{self.W_E.weight.shape=}")
        self.W_E.weight.data[: self.n_inputs, :] = torch.eye(self.n_inputs)
        self.W_E.weight.data[self.n_inputs :, :] = torch.zeros(
            self.d_embed - self.n_inputs, self.n_inputs
        )

        assert self.n_outputs == 1, "Only one output supported"
        out_idx = detailled_circuit[-1][3]
        self.W_U.weight.data = torch.zeros(self.n_outputs, self.d_embed)
        self.W_U.weight.data[0, out_idx] = 1.0

        assert max(x[4] for x in detailled_circuit) < self.n_layers, "Not enough layers"
        for i in range(self.n_layers):
            # torch.nn shapes are (d_output, d_input)
            # linear1.shape = (d_mlp, d_embed)
            # linear2.shape = (d_embed, d_mlp)
            self.layers[i].linear1.weight.data = torch.zeros(self.d_mlp, self.d_embed)
            self.layers[i].linear1.bias.data = torch.zeros(self.d_mlp)
            self.layers[i].linear2.weight.data = torch.zeros(self.d_embed, self.d_mlp)

        used_neurons = [0 for _ in range(self.n_layers)]
        for i, (gate, arg1, arg2, out_idx, min_layer) in enumerate(detailled_circuit):
            if gate == "AND":
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
        print("Used neurons per layer:", used_neurons)
