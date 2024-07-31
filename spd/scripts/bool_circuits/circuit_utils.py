import random
from typing import Literal

import sympy
import torch
from graphviz import Digraph
from jaxtyping import Float
from torch import Tensor

from spd.log import logger

OPERATIONS = ["AND", "OR", "NOT"]


class BooleanOperation:
    def __init__(self, op: Literal["AND", "OR", "NOT"], arg1: int, arg2: int | None) -> None:
        self.op_name: Literal["AND", "OR", "NOT"] = op
        self.arg1: int = arg1
        self.arg2: int | None = arg2
        self.out_idx: int | None = None
        self.min_layer_needed: int | None = None

    def __call__(self, values: list[int]) -> int:
        if self.op_name == "NOT":
            return 1 - values[self.arg1]
        elif self.op_name == "AND":
            assert self.arg2 is not None
            return values[self.arg1] & values[self.arg2]
        elif self.op_name == "OR":
            assert self.arg2 is not None
            return values[self.arg1] | values[self.arg2]
        else:
            raise ValueError(f"Unknown operation: {self.op_name}")

    def to_sympy(self, variables: list[sympy.Symbol]) -> sympy.logic.boolalg.BooleanFunction:
        if self.op_name == "NOT":
            return sympy.Not(variables[self.arg1])
        elif self.op_name == "AND":
            assert self.arg2 is not None
            return sympy.And(variables[self.arg1], variables[self.arg2])
        elif self.op_name == "OR":
            assert self.arg2 is not None
            return sympy.Or(variables[self.arg1], variables[self.arg2])
        else:
            raise ValueError(f"Unknown operation: {self.op_name}")

    def __repr__(self) -> str:
        return f"{self.op_name}({self.arg1}, {self.arg2})"


def create_circuit_str(circuit: list[BooleanOperation], n_inputs: int) -> str:
    """Create string repr of circuit using sympy"""
    inputs = sympy.symbols(f"x:{n_inputs}")

    outputs = list(inputs)
    for operation in circuit:
        outputs.append(operation.to_sympy(outputs))

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
                input1 = rng.randint(idx_range[0], idx_range[1])
                circuit.append(BooleanOperation(op, input1, None))
            elif op in ["AND", "OR"]:
                input1 = rng.randint(idx_range[0], idx_range[1])
                input2 = rng.randint(idx_range[0], idx_range[1])
                while input2 == input1:
                    input2 = rng.randint(idx_range[0], idx_range[1])
                circuit.append(BooleanOperation(op, input1, input2))
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


def evaluate_circuit(inputs: list[int], circuit: list[BooleanOperation]) -> int:
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
            layer1 = 0 if op.arg1 < n_inputs else circuit[op.arg1 - n_inputs].min_layer_needed
            layer2 = (
                0
                if op.arg2 is None or op.arg2 < n_inputs
                else circuit[op.arg2 - n_inputs].min_layer_needed
            )
            # Continue if we haven't determined the layer for one of the inputs *yet*
            if layer1 is None or layer2 is None:
                continue
            # Set the layer at which this gate's output is available
            circuit[i].min_layer_needed = max(layer1, layer2) + 1
        # Check if the algorithm has convered
        fully_specificed = all([op.min_layer_needed is not None for op in circuit])

    return circuit


def plot_circuit(circuit: list[BooleanOperation], num_inputs: int, filename: str | None = None):
    dot = Digraph(comment="Boolean Circuit")
    dot.attr(rankdir="TB")

    # Add input nodes
    for i in range(num_inputs):
        dot.node(f"x{i}", f"x{i}")

    # Add operation nodes
    for i, op in enumerate(circuit):
        op_id = f"op{i}"
        label = f"[{i}] {op.op_name}"
        if op.out_idx is not None:
            label += f"\nout_idx: {op.out_idx}"
        dot.node(op_id, label)

        # Connect inputs to this operation
        dot.edge(f"x{op.arg1}" if op.arg1 < num_inputs else f"op{op.arg1 - num_inputs}", op_id)
        if op.arg2 is not None:
            dot.edge(f"x{op.arg2}" if op.arg2 < num_inputs else f"op{op.arg2 - num_inputs}", op_id)

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
