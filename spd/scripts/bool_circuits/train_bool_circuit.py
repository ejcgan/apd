"""Trains a neural network to implement a boolean circuit."""

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


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    global_seed: int = 0
    circuit_seed: int = 0
    num_inputs: int
    num_operations: int
    hidden_size: int
    num_layers: int
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    out_file: RootPath


OPERATIONS = ["AND", "OR", "XOR", "NOT"]

YesOperation = tuple[Literal["AND", "OR", "XOR"], int, int]
NotOperation = tuple[Literal["NOT"], int, None]

Operation = YesOperation | NotOperation


class BooleanCircuitDataset(Dataset[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]]):
    def __init__(
        self, num_inputs: int, num_operations: int, circuit_seed: int, num_samples: int = 10000
    ):
        self.num_inputs = num_inputs
        self.num_operations = num_operations
        self.num_samples = num_samples
        self.circuit_seed = circuit_seed

        self.circuit = self.generate_circuit()

    def generate_circuit(self) -> list[Operation]:
        operations: list[Literal["AND", "OR", "XOR", "NOT"]] = ["AND", "OR", "XOR", "NOT"]
        circuit: list[Operation] = []
        rng = random.Random(self.circuit_seed)

        for i in range(self.num_operations):
            op = rng.choice(operations)
            if op == "NOT":
                input1 = rng.randint(0, self.num_inputs + i - 1)
                not_tup: Operation = ("NOT", input1, None)
                circuit.append(not_tup)
            else:
                available_inputs = list(range(self.num_inputs + i))
                input1 = rng.choice(available_inputs)
                available_inputs.remove(input1)
                input2 = rng.choice(available_inputs)
                op_tup: Operation = (op, input1, input2)
                circuit.append(op_tup)
        return circuit

    def evaluate_circuit(self, inputs: list[int]) -> int:
        values = inputs.copy()

        for op, input1, input2 in self.circuit:
            if op == "AND":
                assert input2 is not None
                result = values[input1] & values[input2]
            elif op == "OR":
                assert input2 is not None
                result = values[input1] | values[input2]
            elif op == "XOR":
                assert input2 is not None
                result = values[input1] ^ values[input2]
            elif op == "NOT":
                result = 1 - values[input1]
            else:
                raise ValueError(f"Unknown operation: {op}")
            values.append(result)

        return values[-1]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]:
        inputs = [random.randint(0, 1) for _ in range(self.num_inputs)]
        output = self.evaluate_circuit(inputs)
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)


class BoolCircuitModel(nn.Module):
    def __init__(self, num_inputs: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_layer = nn.Linear(num_inputs, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: Float[Tensor, "batch inputs"]) -> Float[Tensor, " batch"]:
        residual = self.input_layer(x)
        for layer in self.layers:
            residual = residual + F.relu(layer(residual))
        return self.output_layer(residual).squeeze(-1)


def train(
    config: Config,
    model: BoolCircuitModel,
    dataloader: DataLoader[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]],
    device: str,
) -> None:
    Path(config.out_file).parent.mkdir(parents=True, exist_ok=True)

    set_seed(config.global_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    progress_bar = tqdm(range(config.steps), desc="Training")
    for step in progress_bar:
        total_loss = 0.0
        for batch_inputs, batch_outputs in dataloader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = F.binary_cross_entropy_with_logits(predictions, batch_outputs.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if step % config.print_freq == 0:
            tqdm.write(f"Step {step}: loss={avg_loss:.4f}")

    torch.save(model.state_dict(), config.out_file)
    logger.info(f"Saved model to {config.out_file}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(
        global_seed=0,
        circuit_seed=0,
        num_inputs=3,
        num_operations=5,
        hidden_size=12,
        num_layers=2,
        batch_size=8,
        steps=1000,
        print_freq=100,
        lr=0.001,
        out_file="spd/scripts/bool_circuits/out/bool_circuit_model.pt",  # pyright: ignore [reportArgumentType]
    )
    logger.info(f"Config: {config}")
    dataset = BooleanCircuitDataset(
        num_inputs=config.num_inputs,
        num_operations=config.num_operations,
        circuit_seed=config.circuit_seed,
    )
    logger.info(f"Circuit: n_inputs={dataset.num_inputs} - {dataset.circuit}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = BoolCircuitModel(config.num_inputs, config.hidden_size, config.num_layers).to(device)

    train(
        config=config,
        model=model,
        dataloader=dataloader,
        device=device,
    )
