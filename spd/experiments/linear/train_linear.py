"""Trains a deep linear model on one-hot input vectors."""

from pathlib import Path

import torch
import wandb
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch.nn import functional as F

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import DeepLinearModel
from spd.types import RootPath
from spd.utils import DatasetGeneratedDataLoader, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    n_features: PositiveInt
    n_layers: PositiveInt
    n_instances: PositiveInt
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    out_file: RootPath | None = None


def train(
    config: Config,
    model: DeepLinearModel,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float | None:
    if config.out_file is not None:
        Path(config.out_file).parent.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    final_loss = None
    for step, (batch, labels) in enumerate(dataloader):
        if step >= config.steps:
            break
        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)
        loss = F.mse_loss(model(batch), labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if step % config.print_freq == 0:
            print(f"Step {step}: loss={final_loss}")

    if config.out_file is not None:
        torch.save(model.state_dict(), config.out_file)
        print(f"Saved model to {config.out_file}")
    return final_loss


if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    config = Config(
        n_features=5,
        n_layers=3,
        n_instances=2,
        batch_size=100,
        steps=1000,
        print_freq=100,
        lr=0.01,
        out_file="spd/experiments/linear/out/linear.pt",  # pyright: ignore [reportArgumentType]
    )
    model = DeepLinearModel(config.n_features, config.n_layers, config.n_instances).to(device)
    dataset = DeepLinearDataset(config.n_features, config.n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    train(config, model, dataloader, device)

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
