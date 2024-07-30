import random
from typing import Any, Literal

import sympy
import torch
from jaxtyping import Float
from torch import Tensor

from spd.log import logger

OPERATIONS = ["AND", "OR", "NOT"]
NotOperation = tuple[Literal["NOT"], int, None]
DetailedNotOperation = tuple[Literal["NOT"], int, None, int, int]
TwoArgOperation = tuple[Literal["AND", "OR"], int, int]
DetailedTwoArgOperation = tuple[Literal["AND", "OR"], int, int, int, int]
Operation = NotOperation | TwoArgOperation
DetailedOperation = DetailedNotOperation | DetailedTwoArgOperation
Circuit = list[Operation]
DetailedCircuit = list[DetailedOperation]


def generate_circuit(
    n_inputs: int,
    n_operations: int,
    circuit_seed: int,
    truth_range: tuple[float, float],
    circuit_min_variables: int,
    max_tries: int = 100,
) -> list[TwoArgOperation | NotOperation]:
    rng = random.Random(circuit_seed)

    for n_attempts in range(max_tries):
        circuit: list[NotOperation | TwoArgOperation] = []

        for i in range(n_operations):
            if i == n_operations - 1:
                op = rng.choice([o for o in OPERATIONS if o != "NOT"])
            else:
                op = rng.choice(OPERATIONS)
            # Always use a non-original input for the last half of the operations
            if i >= n_operations / 2:
                idx_range = (n_inputs, n_inputs + i - 1)
            else:
                idx_range = (0, n_inputs + i - 1)
            if op == "NOT":
                input1 = rng.randint(idx_range[0], idx_range[1])
                not_tup: NotOperation = ("NOT", input1, None)
                circuit.append(not_tup)
            elif op in ["AND", "OR"]:
                input1 = rng.randint(idx_range[0], idx_range[1])
                input2 = rng.randint(idx_range[0], idx_range[1])
                while input2 == input1:
                    input2 = rng.randint(idx_range[0], idx_range[1])
                circuit.append((op, input1, input2))  # pyright: ignore [reportArgumentType]
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


def create_truth_table(
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
        [evaluate_circuit(inputs.tolist(), circuit) for inputs in all_possible_inputs],
    ).unsqueeze(1)
    return torch.cat([all_possible_inputs, outputs], dim=-1).type(torch.get_default_dtype())


def create_circuit_str(circuit: list[TwoArgOperation | NotOperation], n_inputs: int) -> str:
    """Create string repr of circuit using sympy"""
    inputs = sympy.symbols(f"x:{n_inputs}")

    def apply_gate(
        gate_type: Literal["NOT", "AND", "OR"], *args: tuple[sympy.Expr]
    ) -> sympy.logic.boolalg.BooleanFunction:
        if gate_type == "NOT":
            return sympy.Not(args[0])
        elif gate_type == "AND":
            return sympy.And(*args)
        elif gate_type == "OR":
            return sympy.Or(*args)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    outputs = list(inputs)
    for gate_type, *connections in circuit:
        gate_inputs = [outputs[conn] for conn in connections if conn is not None]
        gate_output = apply_gate(gate_type, *gate_inputs)
        outputs.append(gate_output)

    return str(outputs[-1])


def make_detailed_circuit(base_circuit: Circuit, n_inputs: int) -> DetailedCircuit:
    """Adds (i) the output index and (ii) the minimum layer to implement the gate, to the circuit"""
    # New format for the circuit: (gate, arg1, arg2, out_idx, min_layer_needed)
    circuit: list[Any] = []

    # Assign an output index to each gate
    for i, (gate, arg1, arg2) in enumerate(base_circuit):
        op = [gate, arg1, arg2, n_inputs + i, None]
        circuit.append(op)

    # Find the minimum layer needed for each gate
    fully_specificed = False
    while not fully_specificed:
        for i, (_, arg1, arg2, _, min_layer_needed) in enumerate(circuit):
            # Continue if we have already determined the layer for this gate
            if min_layer_needed is not None:
                continue
            # Determine the layer at which the inputs are available.
            # Note that circuit[...][4] can be None
            layer1 = 0 if arg1 < n_inputs else circuit[arg1 - n_inputs][4]
            layer2 = 0 if arg2 is None or arg2 < n_inputs else circuit[arg2 - n_inputs][4]
            # Continue if we haven't determined the layer for one of the inputs *yet*
            if layer1 is None or layer2 is None:
                continue
            # Set the layer at which this gate's output is available
            circuit[i][4] = max(layer1, layer2) + 1
        # Check if the algorithm has convered
        fully_specificed = all([x[4] is not None for x in circuit])

    out_circuit: DetailedCircuit = []
    for gate, arg1, arg2, out_idx, min_layer_needed in circuit:
        if arg2 is None:
            not_tup: DetailedNotOperation = (gate, arg1, arg2, out_idx, min_layer_needed)
            out_circuit.append(not_tup)
        else:
            two_arg_tup: DetailedTwoArgOperation = (gate, arg1, arg2, out_idx, min_layer_needed)
            out_circuit.append(two_arg_tup)
    return out_circuit
