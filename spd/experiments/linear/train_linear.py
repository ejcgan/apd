"""Trains a deep linear model on one-hot input vectors."""

import json
from pathlib import Path

import torch
import wandb
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch.nn import functional as F

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import DeepLinearModel
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


def train(
    config: Config,
    model: DeepLinearModel,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    out_dir: Path | None = None,
) -> float | None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

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

    if out_dir is not None:
        model_path = out_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        config_path = out_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=4)
        print(f"Saved config to {config_path}")

    return final_loss


if __name__ == "__main__":
    device = "cpu"
    config = Config(
        n_features=5,
        n_layers=3,
        n_instances=6,
        batch_size=100,
        steps=1000,
        print_freq=100,
        lr=0.01,
    )

    set_seed(config.seed)
    # Create a run name based on important config parameters
    run_name = (
        f"linear_n-features{config.n_features}_n-layers{config.n_layers}_"
        f"n-instances{config.n_instances}_seed{config.seed}"
    )
    out_dir = Path(__file__).parent / "out" / run_name

    model = DeepLinearModel(config.n_features, config.n_layers, config.n_instances).to(device)
    dataset = DeepLinearDataset(config.n_features, config.n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    train(config=config, model=model, dataloader=dataloader, device=device, out_dir=out_dir)

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
