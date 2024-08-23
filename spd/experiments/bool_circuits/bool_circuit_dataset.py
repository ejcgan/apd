from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

from spd.experiments.bool_circuits.bool_circuit_utils import (
    BooleanOperation,
    create_truth_table,
)


class BooleanCircuitDataset(Dataset[tuple[Float[Tensor, " inputs"], Float[Tensor, ""]]]):
    def __init__(
        self, circuit: list[BooleanOperation], n_inputs: int, valid_idxs: list[int] | None = None
    ):
        self.circuit = circuit
        self.n_inputs = n_inputs
        self.valid_idxs = valid_idxs
        self.data_table = create_truth_table(n_inputs, circuit)

    def __len__(self) -> int:
        return len(self.valid_idxs) if self.valid_idxs is not None else 2**self.n_inputs

    def __getitem__(self, idx: int) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
        data_idx = self.valid_idxs[idx] if self.valid_idxs is not None else idx
        data = self.data_table[data_idx].detach().clone()
        return data[:-1], data[-1:]
