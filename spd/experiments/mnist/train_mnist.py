"""Train MNIST models for SPD experiments."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from spd.experiments.mnist.models import MNISTModel, MNISTModelConfig
from spd.log import logger
from spd.utils import set_seed

wandb.require("core")


class MNISTTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    mnist_model_config: MNISTModelConfig
    batch_size: PositiveInt
    epochs: PositiveInt
    seed: int = 0
    lr: float
    weight_decay: float = 0.0
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"


def linear_lr(epoch: int, epochs: int) -> float:
    return 1 - (epoch / epochs)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(epoch: int, epochs: int) -> float:
    return np.cos(0.5 * np.pi * epoch / (epochs - 1))


def get_lr_schedule_fn(
    schedule: Literal["constant", "linear", "cosine"],
) -> Callable[[int, int], float]:
    if schedule == "constant":
        return constant_lr
    elif schedule == "linear":
        return linear_lr
    elif schedule == "cosine":
        return cosine_decay_lr
    else:
        raise ValueError(f"Unknown lr schedule: {schedule}")


def train(
    model: MNISTModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    log_wandb: bool,
    epochs: int = 10,
    print_freq: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    lr_schedule: Callable[[int, int], float] = constant_lr,
) -> tuple[list[float], list[float]]:
    """Train a MNIST model."""
    # Initialize metrics
    train_losses = []
    test_accuracies = []

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Adjust learning rate based on schedule
        epoch_lr = lr * lr_schedule(epoch, epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = epoch_lr

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % print_freq == 0:
                pbar.set_postfix({"loss": loss.item(), "lr": epoch_lr})
                if log_wandb:
                    wandb.log(
                        {
                            "batch_loss": loss.item(),
                            "lr": epoch_lr,
                            "batch": batch_idx + epoch * len(train_loader),
                        }
                    )

        # Calculate average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

        if log_wandb:
            wandb.log(
                {"epoch": epoch, "train_loss": avg_train_loss, "test_accuracy": test_accuracy}
            )

    return train_losses, test_accuracies


def evaluate(model: MNISTModel, test_loader: DataLoader, device: str) -> float:
    """Evaluate a MNIST model on the test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def plot_training_metrics(
    train_losses: list[float], test_accuracies: list[float], filepath: Path
) -> None:
    """Plot training metrics and save the figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training loss
    ax1.plot(train_losses, "b-")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot test accuracy
    ax2.plot(test_accuracies, "r-")
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def run_train(config: MNISTTrainConfig, device: str) -> None:
    """Run training process for MNIST model."""
    # Set random seed
    set_seed(config.seed)

    # Prepare data loaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create model
    model = MNISTModel(config=config.mnist_model_config)
    model.to(device)

    # Generate run name
    model_cfg = config.mnist_model_config
    run_name = (
        f"mnist_layers{model_cfg.n_layers}_hidden{model_cfg.hidden_dim}_"
        f"bs{config.batch_size}_lr{config.lr}_seed{config.seed}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_dir = Path(__file__).parent / "out" / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if config.wandb_project:
        wandb.init(project=config.wandb_project, name=run_name)

    # Save config
    config_path = out_dir / "mnist_train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, indent=2)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    # Train model
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)
    train_losses, test_accuracies = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        log_wandb=config.wandb_project is not None,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_schedule=lr_schedule_fn,
    )

    # Save model
    model_path = out_dir / "mnist_model.pth"
    torch.save(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # Plot and save training metrics
    metrics_path = out_dir / "training_metrics.png"
    plot_training_metrics(train_losses, test_accuracies, metrics_path)
    logger.info(f"Saved training metrics plot to {metrics_path}")
    if config.wandb_project:
        wandb.log({"training_metrics": wandb.Image(metrics_path)})


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1-layer model
    config_1layer = MNISTTrainConfig(
        wandb_project="spd-mnist",
        mnist_model_config=MNISTModelConfig(
            n_layers=1,
            hidden_dim=128,
            device=device,
        ),
        batch_size=128,
        epochs=10,
        seed=0,
        lr=0.001,
    )

    # 2-layer model
    config_2layer = MNISTTrainConfig(
        wandb_project="spd-mnist",
        mnist_model_config=MNISTModelConfig(
            n_layers=2,
            hidden_dim=128,
            device=device,
        ),
        batch_size=128,
        epochs=10,
        seed=0,
        lr=0.001,
    )

    # 3-layer model
    config_3layer = MNISTTrainConfig(
        wandb_project="spd-mnist",
        mnist_model_config=MNISTModelConfig(
            n_layers=3,
            hidden_dim=128,
            device=device,
        ),
        batch_size=128,
        epochs=10,
        seed=0,
        lr=0.001,
    )

    # Train all models
    for config in [config_1layer, config_2layer, config_3layer]:
        logger.info(f"Training {config.mnist_model_config.n_layers}-layer model")
        run_train(config, device)
