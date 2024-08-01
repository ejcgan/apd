from collections.abc import Iterator

import torch
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DeepLinearDataset(Dataset[Float[Tensor, "n_instances n_features"]]):
    def __init__(self, n_features: int, n_instances: int):
        self.n_features = n_features
        self.n_instances = n_instances

    def __len__(self):
        return 2**31

    def __getitem__(self, _: int) -> Float[Tensor, "n_instances n_features"]:
        # This method will not be used directly
        pass

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "n_instances n_features"], Float[Tensor, "n_instances n_features"]]:
        x_idx = torch.randint(0, self.n_features, (batch_size, self.n_instances))
        x = torch.nn.functional.one_hot(x_idx, num_classes=self.n_features).float()
        return x, x.clone().detach()


class DeepLinearDataLoader(DataLoader[Float[Tensor, "n_instances n_features"]]):
    def __init__(
        self,
        dataset: DeepLinearDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(  # type: ignore
        self,
    ) -> Iterator[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances n_features"],
        ]
    ]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)
