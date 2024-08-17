"""Trains a deep linear model on one-hot input vectors."""

from pathlib import Path

import torch
import wandb
from pydantic import BaseModel, ConfigDict
from torch.nn import functional as F

from spd.models.linear_models import DeepLinearModel
from spd.scripts.linear.linear_dataset import DeepLinearDataset
from spd.types import RootPath
from spd.utils import BatchedDataLoader, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    n_features: int
    n_layers: int
    n_instances: int
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    out_file: RootPath


def train(config: Config):
    Path(config.out_file).parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.seed)
    model = DeepLinearModel(config.n_features, config.n_layers, config.n_instances).to(device)
    dataset = DeepLinearDataset(config.n_features, config.n_instances)
    dataloader = BatchedDataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    data_iter = iter(dataloader)
    for step in range(config.steps):
        optimizer.zero_grad()
        batch, labels = next(data_iter)
        batch = batch.to(device)
        labels = labels.to(device)
        loss = F.mse_loss(model(batch), labels)
        loss.backward()
        optimizer.step()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={loss.item()}")

    # Assert that, for each instance, the multiplication of the params is approximately the identity
    for instance in range(config.n_instances):
        weight_list = [layer[instance] for layer in model.layers]
        # Do a matrix multiplication of all elements in the weight_list. E.g. if weight_list has
        # [a, b, c], then do a @ b @ c
        for j in range(len(weight_list) - 1):
            weight_list[j + 1] = weight_list[j] @ weight_list[j + 1]
        # Assert that the last weight is approximately the identity matrix
        assert torch.allclose(
            weight_list[-1],
            torch.eye(config.n_features, device=weight_list[-1].device),
            atol=1e-5,
        )

    torch.save(model.state_dict(), config.out_file)
    print(f"Saved model to {config.out_file}")


if __name__ == "__main__":
    config = Config(
        n_features=5,
        n_layers=3,
        n_instances=2,
        batch_size=100,
        steps=1000,
        print_freq=100,
        lr=0.01,
        out_file="spd/scripts/linear/out/linear.pt",  # pyright: ignore [reportArgumentType]
    )
    train(config)
