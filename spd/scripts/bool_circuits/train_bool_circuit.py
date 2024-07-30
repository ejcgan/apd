"""Trains a neural network to implement a boolean circuit."""

import json
import random
from pathlib import Path
from typing import Literal

import torch
import wandb
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from spd.log import logger
from spd.types import RootPath
from spd.utils import set_seed

wandb.require("core")
DEFAULT_TORCH_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_TORCH_DTYPE)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    global_seed: int = 0
    circuit_seed: int = 0
    n_inputs: int
    n_operations: int
    hidden_size: int
    n_layers: int
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    out_dir: RootPath | None = Path(__file__).parent / "out"
    truth_range: tuple[float, float]  # [min_percent_1s, max_percent_1s]
    eval_pct: float = 0.3
    eval_every_n_samples: int = 1000


OPERATIONS = ["AND", "OR", "NOT"]

NotOperation = tuple[Literal["NOT"], int, None]
TwoArgOperation = tuple[Literal["AND", "OR"], int, int]


class BooleanCircuit:
    def __init__(
        self,
        n_inputs: int,
        n_operations: int,
        circuit_seed: int,
        truth_range: tuple[float, float],
    ):
        self.n_inputs = n_inputs
        self.n_operations = n_operations
        self.circuit_seed = circuit_seed
        self.truth_range = truth_range

        self.circuit = self.generate_circuit()
        self.data_table = self.truth_table(self.n_inputs, self.circuit)

    def generate_circuit(self, max_tries: int = 100) -> list[TwoArgOperation | NotOperation]:
        rng = random.Random(self.circuit_seed)

        for n_attempts in range(max_tries):
            circuit: list[NotOperation | TwoArgOperation] = []

            for i in range(self.n_operations):
                op = rng.choice(OPERATIONS)
                if op == "NOT":
                    input1 = rng.randint(0, self.n_inputs + i - 1)
                    not_tup: NotOperation = ("NOT", input1, None)
                    circuit.append(not_tup)
                elif op in ["AND", "OR"]:
                    input1 = rng.randint(0, self.n_inputs + i - 1)
                    input2 = rng.randint(0, self.n_inputs + i - 1)
                    while input2 == input1:
                        input2 = rng.randint(0, self.n_inputs + i - 1)
                    circuit.append((op, input1, input2))  # pyright: ignore [reportArgumentType]
                else:
                    raise ValueError(f"Unknown operation: {op}")

            truth_table = BooleanCircuit.truth_table(self.n_inputs, circuit)
            truth_percentage = truth_table[:, -1].type(torch.get_default_dtype()).mean().item()

            if self.truth_range[0] <= truth_percentage <= self.truth_range[1]:
                logger.info(f"Generated circuit in {n_attempts + 1} attempts")
                return circuit

        raise ValueError(
            f"Failed to generate a circuit within the specified truth range after {max_tries + 1} "
            f"attempts."
        )

    @staticmethod
    def evaluate_circuit(inputs: list[int], circuit: list[TwoArgOperation | NotOperation]) -> int:
        values = inputs.copy()

        for op, input1, input2 in circuit:
            if op == "AND":
                assert input2 is not None
                result = values[input1] & values[input2]
            elif op == "OR":
                assert input2 is not None
                result = values[input1] | values[input2]
            elif op == "NOT":
                result = 1 - values[input1]
            else:
                raise ValueError(f"Unknown operation: {op}")
            values.append(result)

        return values[-1]

    @staticmethod
    def truth_table(
        n_inputs: int, circuit: list[TwoArgOperation | NotOperation]
    ) -> Float[Tensor, "all_possible_inputs inputs+1"]:
        """Get the truth table for the circuit.

        Returns a tensor of shape (2**n_inputs, n_inputs + 1) where the final
        column is the output of the circuit.
        """
        # Get all combinations of boolean inputs
        n_input_combinations = 2**n_inputs
        all_possible_inputs = torch.tensor(
            [list(map(int, bin(i)[2:].zfill(n_inputs))) for i in range(n_input_combinations)],
        )
        outputs = torch.tensor(
            [
                BooleanCircuit.evaluate_circuit(inputs.tolist(), circuit)
                for inputs in all_possible_inputs
            ],
        ).unsqueeze(1)
        return torch.cat([all_possible_inputs, outputs], dim=-1).type(torch.get_default_dtype())

    def __str__(self) -> str:
        return f"Circuit: n_inputs={self.n_inputs} - {self.circuit}"


class BooleanCircuitDataset(Dataset[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]]):
    def __init__(
        self,
        bool_circuit: BooleanCircuit,
        valid_idxs: list[int],
    ):
        self.bool_circuit = bool_circuit
        self.valid_idxs = valid_idxs

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        data_idx = self.valid_idxs[idx]
        data = self.bool_circuit.data_table[data_idx].detach().clone()
        return data[:-1], data[-1:]


class MLP(nn.Module):
    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.linear1 = nn.Linear(d_embed, d_mlp)
        self.linear2 = nn.Linear(d_mlp, d_embed)

    def forward(self, x: Float[Tensor, "... d_embed"]) -> Float[Tensor, "... d_embed"]:
        return self.linear2(F.relu(self.linear1(x)))


class Transformer(nn.Module):
    def __init__(self, n_inputs: int, d_embed: int, d_mlp: int, n_layers: int, n_outputs: int = 1):
        super().__init__()
        self.W_E = nn.Linear(n_inputs, d_embed)
        self.W_U = nn.Linear(d_embed, n_outputs)
        self.layers = nn.ModuleList([MLP(d_embed, d_mlp) for _ in range(n_layers)])

    def forward(self, x: Float[Tensor, "batch inputs"]) -> Float[Tensor, "batch outputs"]:
        residual = self.W_E(x)
        for layer in self.layers:
            residual = residual + layer(residual)
        return self.W_U(residual)


def evaluate(
    model: Transformer,
    dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = F.binary_cross_entropy_with_logits(
                predictions, labels.type(torch.get_default_dtype())
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(
    config: Config,
    model: Transformer,
    train_dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    eval_dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: str,
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
            eval_loss = evaluate(model, eval_dataloader, device)
            tqdm.write(f"Step {step}: loss={loss.item():.4f}, eval_loss={eval_loss:.4f}")
        elif step % config.print_freq == 0:
            tqdm.write(f"Step {step}: loss={loss.item():.4f}")

    if config.out_dir is not None:
        # Save config and model
        exp_info = (
            f"inp{config.n_inputs}-op{config.n_operations}-hid{config.hidden_size}-"
            f"lay{config.n_layers}-circseed{config.circuit_seed}-seed{config.global_seed}"
        )
        experiment_dir = config.out_dir / exp_info
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(experiment_dir / "config.json", "w") as f:
            json.dump(config.model_dump(), f, indent=4)
        torch.save(model.state_dict(), experiment_dir / "model.pt")
        logger.info(f"Saved model to {experiment_dir / 'model.pt'}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(
        global_seed=0,
        circuit_seed=0,
        n_inputs=10,
        n_operations=20,
        hidden_size=8,
        n_layers=2,
        batch_size=16,
        steps=5000,
        print_freq=500,
        lr=0.001,
        truth_range=(0.4, 0.6),
        eval_pct=0.2,
        eval_every_n_samples=500,
    )
    logger.info(f"Config: {config}")
    bool_circuit = BooleanCircuit(
        n_inputs=config.n_inputs,
        n_operations=config.n_operations,
        circuit_seed=config.circuit_seed,
        truth_range=config.truth_range,
    )
    logger.info(bool_circuit)
    logger.info(f"Truth table:\n{bool_circuit.truth_table(config.n_inputs, bool_circuit.circuit)}")
    # Randomly select eval_pct idxs from range(len(bool_circuit.data_table))
    n_input_combinations = 2**config.n_inputs
    assert n_input_combinations == len(bool_circuit.data_table)
    eval_idxs = random.sample(
        range(n_input_combinations), int(config.eval_pct * n_input_combinations)
    )
    train_idxs = [i for i in range(n_input_combinations) if i not in eval_idxs]
    train_dataset = BooleanCircuitDataset(bool_circuit, valid_idxs=train_idxs)
    eval_dataset = BooleanCircuitDataset(bool_circuit, valid_idxs=eval_idxs)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    model = Transformer(
        n_inputs=config.n_inputs,
        d_embed=config.hidden_size,
        d_mlp=config.hidden_size,
        n_layers=config.n_layers,
    ).to(device)

    train(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        device=device,
    )
