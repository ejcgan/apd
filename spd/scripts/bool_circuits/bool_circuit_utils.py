import random
from abc import ABC, abstractmethod
from typing import Any, Literal

import sympy
import torch
from graphviz import Digraph
from jaxtyping import Float
from sympy.core.symbol import Symbol
from torch import Tensor

from spd.log import logger

OPERATIONS = ["AND", "OR", "NOT"]


class BooleanOperation(ABC):
    name: str

    def __init__(
        self,
        input_idx1: int,
        input_idx2: int | None = None,
        out_idx: int | None = None,
        min_layer_needed: int | None = None,
    ) -> None:
        self.input_idx1: int = input_idx1
        self.input_idx2: int | None = input_idx2
        self.out_idx: int | None = out_idx
        self.min_layer_needed: int | None = min_layer_needed

    @abstractmethod
    def __call__(self, inputs: list[bool]) -> bool:
        pass

    @abstractmethod
    def call_sympy(self, inputs: list[Symbol]) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.name}({self.input_idx1}, {self.input_idx2})"


class AndOperation(BooleanOperation):
    name = "AND"

    def __call__(self, inputs: list[bool]) -> bool:
        assert self.input_idx2 is not None
        return inputs[self.input_idx1] & inputs[self.input_idx2]

    def call_sympy(self, inputs: list[Symbol]) -> Any:
        assert self.input_idx2 is not None
        return sympy.And(inputs[self.input_idx1], inputs[self.input_idx2])


class OrOperation(BooleanOperation):
    name = "OR"

    def __call__(self, inputs: list[bool]) -> bool:
        assert self.input_idx2 is not None
        return inputs[self.input_idx1] | inputs[self.input_idx2]

    def call_sympy(self, inputs: list[Symbol]) -> Any:
        assert self.input_idx2 is not None
        return sympy.Or(inputs[self.input_idx1], inputs[self.input_idx2])


class NotOperation(BooleanOperation):
    name = "NOT"

    def __call__(self, inputs: list[bool]) -> bool:
        assert self.input_idx2 is None
        return not inputs[self.input_idx1]

    def call_sympy(self, inputs: list[Symbol]) -> Any:
        return sympy.Not(inputs[self.input_idx1])


def create_circuit_str(circuit: list[BooleanOperation], n_inputs: int) -> str:
    """Create string repr of circuit using sympy"""
    inputs = sympy.symbols(f"x:{n_inputs}")

    outputs = list(inputs)
    for operation in circuit:
        outputs.append(operation.call_sympy(outputs))

    return str(outputs[-1])


def generate_circuit(
    n_inputs: int,
    n_operations: int,
    circuit_seed: int,
    truth_range: tuple[float, float],
    circuit_min_variables: int,
    max_tries: int = 100,
) -> list[BooleanOperation]:
    rng = random.Random(circuit_seed)

    for n_attempts in range(max_tries):
        circuit: list[BooleanOperation] = []
        for i in range(n_operations):
            if i == n_operations - 1:
                rng_out = rng.choice([o for o in OPERATIONS if o != "NOT"])
                assert rng_out == "AND" or rng_out == "OR"
                op: Literal["AND", "OR", "NOT"] = rng_out
            else:
                rng_out = rng.choice(OPERATIONS)
                assert rng_out == "AND" or rng_out == "OR" or rng_out == "NOT"
                op: Literal["AND", "OR", "NOT"] = rng_out
            # Always use a non-original input for the last half of the operations
            if i >= n_operations / 2:
                idx_range = (n_inputs, n_inputs + i - 1)
            else:
                idx_range = (0, n_inputs + i - 1)
            if op == "NOT":
                input_idx1 = rng.randint(idx_range[0], idx_range[1])
                circuit.append(NotOperation(input_idx1=input_idx1, input_idx2=None))
            elif op in ["AND", "OR"]:
                input_idx1 = rng.randint(idx_range[0], idx_range[1])
                input_idx2 = rng.randint(idx_range[0], idx_range[1])
                while input_idx2 == input_idx1:
                    input_idx2 = rng.randint(idx_range[0], idx_range[1])
                if op == "AND":
                    circuit.append(AndOperation(input_idx1, input_idx2))
                elif op == "OR":
                    circuit.append(OrOperation(input_idx1, input_idx2))
            else:
                raise ValueError(f"Unknown operation: {op}")

        truth_table = create_truth_table(n_inputs, circuit)
        truth_percentage = truth_table[:, -1].type(torch.get_default_dtype()).mean().item()

        if truth_range[0] <= truth_percentage <= truth_range[1]:
            circuit_str = create_circuit_str(circuit, n_inputs)
            # Check that there are at least `n_inputs` variables in the final circuit
            # Count the number of "x" in the circuit string.
            if circuit_str.count("x") >= circuit_min_variables:
                logger.info(f"Generated circuit in {n_attempts + 1} attempts")
                return circuit

    raise ValueError(
        f"Failed to generate a circuit within the specified truth range after {max_tries + 1} "
        f"attempts."
    )


def evaluate_circuit(inputs: list[bool], circuit: list[BooleanOperation]) -> bool:
    values = inputs.copy()

    for operation in circuit:
        result = operation(values)
        values.append(result)

    return values[-1]


def create_truth_table(
    n_inputs: int, circuit: list[BooleanOperation]
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
        [evaluate_circuit(inputs.tolist(), circuit) for inputs in all_possible_inputs],
    ).unsqueeze(1)
    return torch.cat([all_possible_inputs, outputs], dim=-1).type(torch.get_default_dtype())


def make_detailed_circuit(circuit: list[BooleanOperation], n_inputs: int) -> list[BooleanOperation]:
    """Adds (i) the output index and (ii) the minimum layer to implement the gate, to the circuit"""
    # New format for the circuit: (gate, arg1, arg2, out_idx, min_layer_needed)

    # Assign an output index to each gate
    for i in range(len(circuit)):
        circuit[i].out_idx = n_inputs + i

    # Find the minimum layer needed for each gate
    fully_specificed = False
    while not fully_specificed:
        for i, op in enumerate(circuit):
            # Continue if we have already determined the layer for this gate
            if op.min_layer_needed is not None:
                continue
            # Determine the layer at which the inputs are available.
            # Note that circuit[...][4] can be None
            layer1 = (
                0
                if op.input_idx1 < n_inputs
                else circuit[op.input_idx1 - n_inputs].min_layer_needed
            )
            layer2 = (
                0
                if op.input_idx2 is None or op.input_idx2 < n_inputs
                else circuit[op.input_idx2 - n_inputs].min_layer_needed
            )
            # Continue if we haven't determined the layer for one of the inputs *yet*
            if layer1 is None or layer2 is None:
                continue
            # Set the layer at which this gate's output is available
            circuit[i].min_layer_needed = max(layer1, layer2) + 1
        # Check if the algorithm has convered
        fully_specificed = all([op.min_layer_needed is not None for op in circuit])

    return circuit


def plot_circuit(
    circuit: list[BooleanOperation],
    num_inputs: int,
    filename: str | None = None,
    show_out_idx: bool = False,
) -> None:
    """Plot the boolean circuit using graphviz.

    Args:
        circuit: The boolean circuit to plot.
        num_inputs: The number of input nodes.
        filename: The filename to save the plot to. If None, the plot is rendered in the notebook.
        show_out_idx: [Only relevant for hand-coded circuits!] Show the residual stream index
            assigned to the output of each operation.
    """
    dot = Digraph(comment="Boolean Circuit")
    dot.attr(rankdir="TB")

    # Add input nodes
    for i in range(num_inputs):
        label = f"[{i}] x{i}"
        if show_out_idx:
            label += f"\nout_idx: {i}"
        dot.node(f"x{i}", label)

    # Add operation nodes
    for i, op in enumerate(circuit):
        op_id = f"op{i}"
        label = f"[{i}] {op.name}"
        if show_out_idx:
            label += f"\nout_idx: {op.out_idx if op.out_idx is not None else '?'}"
        dot.node(op_id, label)

        # Connect inputs to this operation
        dot.edge(
            f"x{op.input_idx1}"
            if op.input_idx1 < num_inputs
            else f"op{op.input_idx1 - num_inputs}",
            op_id,
        )
        if op.input_idx2 is not None:
            dot.edge(
                f"x{op.input_idx2}"
                if op.input_idx2 < num_inputs
                else f"op{op.input_idx2 - num_inputs}",
                op_id,
            )

    # Connect the last operation to the output
    dot.node("output", "Output")
    dot.edge(f"op{len(circuit) - 1}", "output")

    if filename is None:
        # Render the graph in the notebook
        from IPython.display import Image, display

        png_data = dot.pipe(format="png")
        display(Image(png_data))
    else:
        dot.render(filename, format="png", cleanup=True)
        logger.info(f"Saved circuit plot as {filename}")


def form_circuit(
    circuit_repr: list[tuple[Literal["AND", "OR", "NOT"], int, int | None]],
) -> list[BooleanOperation]:
    circuit: list[BooleanOperation] = []
    for op_name, input_idx1, input_idx2 in circuit_repr:
        if op_name == "AND":
            circuit.append(AndOperation(input_idx1, input_idx2))
        elif op_name == "OR":
            circuit.append(OrOperation(input_idx1, input_idx2))
        elif op_name == "NOT":
            circuit.append(NotOperation(input_idx1, None))
    return circuit


def form_circuit_repr(
    circuit: list[BooleanOperation],
) -> list[tuple[Literal["AND", "OR", "NOT"], int, int | None]]:
    circuit_repr: list[tuple[Literal["AND", "OR", "NOT"], int, int | None]] = []
    for op in circuit:
        if isinstance(op, AndOperation):
            circuit_repr.append(("AND", op.input_idx1, op.input_idx2))
        elif isinstance(op, OrOperation):
            circuit_repr.append(("OR", op.input_idx1, op.input_idx2))
        elif isinstance(op, NotOperation):
            circuit_repr.append(("NOT", op.input_idx1, None))
    return circuit_repr
