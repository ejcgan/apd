from pathlib import Path

import torch
from torch import nn


class DeepLinearModel(nn.Module):
    def __init__(self, n_features: int, n_layers: int, n_instances: int):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_instances = n_instances
        self.layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(n_instances, n_features, n_features))
                for _ in range(n_layers)
            ]
        )

        for layer in self.layers:
            nn.init.kaiming_normal_(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.einsum("bif,ifj->bij", x, layer)
        return x

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of inputs. Each input should be a random one-hot vector of dimension
        n_features.
        """
        x_idx = torch.randint(0, self.n_features, (batch_size, self.n_instances))
        x = torch.nn.functional.one_hot(x_idx, num_classes=self.n_features).float()
        return x

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepLinearModel":
        params = torch.load(path)
        # Get the n_features, n_layers, n_instances from the params
        n_layers = len(params.keys())
        n_features = params["layers.0"].shape[1]
        n_instances = params["layers.0"].shape[0]
        model = cls(n_features, n_layers, n_instances)
        model.load_state_dict(params)
        return model


class ParamComponent(nn.Module):
    def __init__(self, n_instances: int, n_features: int, k: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(n_instances, n_features, k))
        self.B = nn.Parameter(torch.empty(n_instances, k, n_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)
        inner_acts = torch.einsum("bif,ifk->bik", x, normed_A)
        out = torch.einsum("bik,ikg->big", inner_acts, self.B)
        return out, inner_acts


class DeepLinearComponentModel(nn.Module):
    def __init__(self, n_features: int, n_layers: int, n_instances: int, k: int | None):
        super().__init__()
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_instances = n_instances
        self.k = k if k is not None else n_features
        self.layers = nn.ModuleList(
            [ParamComponent(n_instances, n_features, self.k) for _ in range(n_layers)]
        )

        for param in self.layers.parameters():
            nn.init.kaiming_normal_(param)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        layer_acts = []
        inner_acts = []
        for layer in self.layers:
            x, inner_act = layer(x)
            layer_acts.append(x)
            inner_acts.append(inner_act)
        return x, layer_acts, inner_acts

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "DeepLinearComponentModel":
        params = torch.load(path)
        n_layers = len(params) // 2
        for param in params:
            assert param.startswith("layers.") and param.endswith(("A", "B"))
        n_instances, n_features, k = params["layers.0.A"].shape

        model = cls(n_features=n_features, n_layers=n_layers, n_instances=n_instances, k=k)
        model.load_state_dict(params)
        return model

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of inputs. Each input should be a random one-hot vector of dimension
        n_features.
        """
        x_idx = torch.randint(0, self.n_features, (batch_size, self.n_instances))
        x = torch.nn.functional.one_hot(x_idx, num_classes=self.n_features)
        return x
