"""Trains a neural network to implement a boolean circuit."""

import json
import random
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.experiments.bool_circuits.bool_circuit_dataset import BooleanCircuitDataset
from spd.experiments.bool_circuits.bool_circuit_utils import (
    BooleanOperation,
    create_circuit_str,
    create_truth_table,
    form_circuit,
    form_circuit_repr,
    generate_circuit,
)
from spd.experiments.bool_circuits.models import BoolCircuitTransformer
from spd.log import logger
from spd.types import RootPath
from spd.utils import set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    global_seed: int = 0
    circuit_seed: int
    n_inputs: int
    n_operations: int
    d_embed: int
    d_mlp: int
    n_layers: int
    n_outputs: int = 1
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    out_dir: RootPath | None = Path(__file__).parent / "out"
    truth_range: tuple[float, float]  # [min_percent_1s, max_percent_1s]
    eval_pct: float = 0.3
    eval_every_n_samples: int = 1000
    circuit_repr: list[tuple[Literal["AND", "OR", "NOT"], int, int | None]] | None = None
    circuit_min_variables: int  # Min number of variables in the final circuit


def evaluate_model(
    model: BoolCircuitTransformer,
    dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: str,
    output_is_logit: bool = True,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            if output_is_logit:
                loss = F.binary_cross_entropy_with_logits(
                    predictions, labels.type(torch.get_default_dtype())
                )
            else:
                loss = F.binary_cross_entropy(predictions, labels.type(torch.get_default_dtype()))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(
    config: Config,
    model: BoolCircuitTransformer,
    train_dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    eval_dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: str,
    circuit_repr: list[tuple[Literal["AND", "OR", "NOT"], int, int | None]],
) -> None:
    set_seed(config.global_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    n_epochs = config.steps / len(train_dataloader)
    logger.info(
        f"steps={config.steps}, dataset_len={len(train_dataloader.dataset)}, "  # pyright: ignore
        f"batch_size={config.batch_size}, n_batches={len(train_dataloader)}, "
        f"n_epochs={n_epochs}"
    )

    progress_bar = tqdm(range(config.steps), desc="Training")
    dataloader_iter = iter(train_dataloader)
    for step in progress_bar:
        try:
            inputs, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            inputs, labels = next(dataloader_iter)

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = F.binary_cross_entropy_with_logits(
            predictions, labels.type(torch.get_default_dtype())
        )
        loss.backward()
        optimizer.step()

        if step % config.eval_every_n_samples == 0 or step == 0 or step == config.steps - 1:
            eval_loss = evaluate_model(model, eval_dataloader, device)
            tqdm.write(f"Step {step}: loss={loss.item():.4f}, eval_loss={eval_loss:.4f}")
        elif step % config.print_freq == 0:
            tqdm.write(f"Step {step}: loss={loss.item():.4f}")

    if config.out_dir is not None:
        # Save config and model
        exp_info = (
            f"inp{config.n_inputs}-op{config.n_operations}-hid{config.d_embed}-"
            f"lay{config.n_layers}-circseed{config.circuit_seed}-seed{config.global_seed}"
        )
        experiment_dir = config.out_dir / exp_info
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(config.model_dump(), f, indent=4)
        torch.save(model.state_dict(), experiment_dir / "model.pt")
        logger.info(f"Saved model to {experiment_dir / 'model.pt'}")

        with open(experiment_dir / "circuit_repr.json", "w") as f:
            json.dump(circuit_repr, f, indent=4)
        logger.info(f"Saved circuit repr to {experiment_dir / 'circuit_repr.json'}")


def get_circuit(config: Config) -> list[BooleanOperation]:
    if config.circuit_repr is None:
        circuit = generate_circuit(
            n_inputs=config.n_inputs,
            n_operations=config.n_operations,
            circuit_seed=config.circuit_seed,
            truth_range=config.truth_range,
            circuit_min_variables=config.circuit_min_variables,
        )
    else:
        circuit = form_circuit(config.circuit_repr)
    return circuit


def get_train_test_dataloaders(
    config: Config, circuit: list[BooleanOperation]
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    # Randomly select eval_pct idxs from range(len(bool_circuit.data_table))
    n_input_combinations = 2**config.n_inputs

    eval_idxs = random.sample(
        range(n_input_combinations), int(config.eval_pct * n_input_combinations)
    )
    train_idxs = [i for i in range(n_input_combinations) if i not in eval_idxs]

    train_dataset = BooleanCircuitDataset(circuit, config.n_inputs, valid_idxs=train_idxs)
    eval_dataset = BooleanCircuitDataset(circuit, config.n_inputs, valid_idxs=eval_idxs)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(
        global_seed=0,
        circuit_seed=1,
        n_inputs=10,
        n_operations=20,
        circuit_min_variables=6,
        d_embed=8,
        d_mlp=8,
        n_layers=1,
        batch_size=16,
        steps=10000,
        print_freq=500,
        lr=0.001,
        truth_range=(0.4, 0.6),
        eval_pct=0.2,
        eval_every_n_samples=500,
    )
    logger.info(f"Config: {config}")

    circuit = get_circuit(config)
    logger.info(f"Circuit: n_inputs={config.n_inputs} - {circuit}")
    logger.info(f"Circuit string: {create_circuit_str(circuit, config.n_inputs)}")

    truth_table = create_truth_table(config.n_inputs, circuit)
    logger.info(f"Truth table:\n{truth_table}")

    train_dataloader, eval_dataloader = get_train_test_dataloaders(config, circuit)
    assert len(truth_table) == 2**config.n_inputs

    model = BoolCircuitTransformer(
        n_inputs=config.n_inputs,
        d_embed=config.d_embed,
        d_mlp=config.d_embed,
        n_layers=config.n_layers,
        n_outputs=config.n_outputs,
    ).to(device)

    train(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
        circuit_repr=form_circuit_repr(circuit),
    )
