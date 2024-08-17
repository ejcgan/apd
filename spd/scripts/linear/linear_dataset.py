import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


class DeepLinearDataset(
    Dataset[tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]]
):
    def __init__(self, n_features: int, n_instances: int):
        self.n_features = n_features
        self.n_instances = n_instances

    def __len__(self):
        return 2**31

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]:
        x_idx = torch.randint(0, self.n_features, (batch_size, self.n_instances))
        x = torch.nn.functional.one_hot(x_idx, num_classes=self.n_features).float()
        return x, x.clone().detach()
