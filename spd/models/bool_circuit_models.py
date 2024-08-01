import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.log import logger
from spd.models.base import Model, SPDModel
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

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BoolCircuitTransformer":
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        params = torch.load(path / "model.pt")

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


class ParamComponents(nn.Module):
    def __init__(self, n_features: int, k: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_features, k))
        self.B = nn.Parameter(torch.empty(k, n_features))

        nn.init.kaiming_normal_(self.A)
        nn.init.kaiming_normal_(self.B)

    def forward(
        self,
        x: Float[Tensor, "... dim"],
    ) -> tuple[Float[Tensor, "... dim"], Float[Tensor, "... k"]]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        inner_acts = torch.einsum("bf,fk->bk", x, normed_A)
        out = torch.einsum("bk,kg->bg", inner_acts, self.B)
        return out, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... dim"],
        topk: int,
        grads: Float[Tensor, "... k"] | None = None,
    ) -> tuple[Float[Tensor, "... dim"], Float[Tensor, "... k"]]:
        """
        Performs a forward pass using only the top-k components.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            grads: Optional gradients for each component

        Returns:
            out: Output tensor
            inner_acts: Component activations
        """
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        inner_acts = torch.einsum("bf,fk->bk", x, normed_A)

        if grads is not None:
            topk_indices = (grads * inner_acts).abs().topk(topk, dim=-1).indices
        else:
            topk_indices = inner_acts.abs().topk(topk, dim=-1).indices

        # Get values in inner_acts corresponding to topk_indices
        topk_values = inner_acts.gather(dim=-1, index=topk_indices)
        inner_acts_topk = torch.zeros_like(inner_acts)
        inner_acts_topk.scatter_(dim=-1, index=topk_indices, src=topk_values)
        out = torch.einsum("bk,kg->bg", inner_acts_topk, self.B)

        return out, inner_acts_topk


class MLPComponents(nn.Module):
    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.linear1 = ParamComponents(d_embed, d_mlp)
        self.linear2 = ParamComponents(d_mlp, d_embed)
        self.bias2 = nn.Parameter(torch.zeros(d_embed))

    def forward(
        self, x: Float[Tensor, "... d_embed"]
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []
        x, inner_acts_linear1 = self.linear1(x)
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        x, inner_acts_linear2 = self.linear2(F.relu(x))
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)
        return x + self.bias2, layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... d_embed"],
        topk: int,
        grads: list[Float[Tensor, "... k"] | None] | None = None,
    ) -> tuple[
        Float[Tensor, "... d_embed"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each linear layer.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            grads: Optional list of gradients for each linear layer

        Returns:
            x: The output of the MLP
            layer_acts: The activations of each linear layer
            inner_acts: The component activations inside each linear layer
        """
        inner_acts = []
        layer_acts = []

        # First linear layer
        grad1 = grads[0] if grads is not None else None
        x, inner_acts_linear1 = self.linear1.forward_topk(x, topk, grad1)
        inner_acts.append(inner_acts_linear1)
        layer_acts.append(x)

        # ReLU activation
        x = F.relu(x)

        # Second linear layer
        grad2 = grads[1] if grads is not None else None
        x, inner_acts_linear2 = self.linear2.forward_topk(x, topk, grad2)
        inner_acts.append(inner_acts_linear2)
        layer_acts.append(x)

        return x + self.bias2, layer_acts, inner_acts


class BoolCircuitSPDTransformer(SPDModel):
    def __init__(self, n_inputs: int, d_embed: int, d_mlp: int, n_layers: int, n_outputs: int = 1):
        super().__init__()
        self.n_inputs = n_inputs
        self.d_embed = d_embed
        self.d_mlp = d_mlp
        self.n_layers = n_layers
        self.n_param_matrices = n_layers * 2
        self.n_outputs = n_outputs

        self.W_E = nn.Linear(n_inputs, d_embed, bias=False)
        self.W_U = nn.Linear(d_embed, n_outputs, bias=False)
        self.layers = nn.ModuleList([MLPComponents(d_embed, d_mlp) for _ in range(n_layers)])

    @property
    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        # Get all B matrices
        all_B_pairs = [
            (self.layers[i].linear1.B, self.layers[i].linear2.B) for i in range(self.n_layers)
        ]
        all_Bs = [B for B_pair in all_B_pairs for B in B_pair]
        return all_Bs

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
            residual, layer_acts_i, inner_acts_i = layer(residual)
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)
        return self.W_U(residual), layer_acts, inner_acts

    def forward_topk(
        self,
        x: Float[Tensor, "... inputs"],
        topk: int,
        all_grads: list[Float[Tensor, "... k"]] | None = None,
    ) -> tuple[
        Float[Tensor, "... outputs"],
        list[Float[Tensor, "... d_embed"] | Float[Tensor, "... d_mlp"]],
        list[Float[Tensor, "... k"]],
    ]:
        """
        Performs a forward pass using only the top-k components for each component activation.

        Args:
            x: Input tensor
            topk: Number of top components to keep
            all_grads: Optional list of gradients for each layer's components

        Returns:
            output: The output of the transformer
            layer_acts: A list of activations for each layer in each MLP
            inner_acts: A list of component activations for each layer in each MLP
        """
        layer_acts = []
        inner_acts = []
        residual = self.W_E(x)

        n_param_matrices_per_layer = self.n_param_matrices // self.n_layers

        for i, layer in enumerate(self.layers):
            # A single layer contains multiple parameter matrices
            layer_grads = (
                all_grads[i * n_param_matrices_per_layer : (i + 1) * n_param_matrices_per_layer]
                if all_grads is not None
                else None
            )
            residual, layer_acts_i, inner_acts_i = layer.forward_topk(residual, topk, layer_grads)
            layer_acts.extend(layer_acts_i)
            inner_acts.extend(inner_acts_i)

        return self.W_U(residual), layer_acts, inner_acts
