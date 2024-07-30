from typing import Literal

from sympy import And, Not, Or, Symbol, simplify, symbols

OPERATIONS = ["AND", "OR", "NOT"]
NotOperation = tuple[Literal["NOT"], int, None]
DetailedNotOperation = tuple[Literal["NOT"], int, None, int, int]
TwoArgOperation = tuple[Literal["AND", "OR"], int, int]
DetailedTwoArgOperation = tuple[Literal["AND", "OR"], int, int, int, int]
Operation = NotOperation | TwoArgOperation
DetailedOperation = DetailedNotOperation | DetailedTwoArgOperation
IncompleteOperation = tuple[Literal["AND", "OR", "NOT"], int, int | None, int, int | None]
Circuit = list[Operation]
DetailedCircuit = list[DetailedOperation]


def evaluate_circuit(inputs: list[int], circuit: Circuit) -> int:
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


def make_detailed_circuit(base_circuit: Circuit, n_inputs: int) -> DetailedCircuit:
    """Adds (i) the output index and (ii) the minimum layer to implement the gate, to the circuit"""
    # New format for the circuit: (gate, arg1, arg2, out_idx, min_layer_needed)
    circuit: list[IncompleteOperation] = []

    # Assign an output index to each gate
    for i, (gate, arg1, arg2) in enumerate(base_circuit):
        op: IncompleteOperation = (gate, arg1, arg2, n_inputs + i, None)
        circuit.append(op)

    # Find the minimum layer needed for each gate
    fully_specificed = False
    while not fully_specificed:
        for i, (gate, arg1, arg2, out_idx, min_layer_needed) in enumerate(circuit):  # noqa: B007
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

    circuit: DetailedCircuit = circuit
    return circuit


def list_circuit_to_sympy(circuit: Circuit, num_inputs: int) -> tuple[Symbol, Symbol]:
    inputs = symbols(f"x:{num_inputs}")

    def apply_gate(gate_type: str, *args) -> Symbol:
        if gate_type == "NOT":
            return Not(args[0])
        elif gate_type == "AND":
            return And(*args)
        elif gate_type == "OR":
            return Or(*args)

    outputs = list(inputs)
    for gate_type, *connections in circuit:
        gate_inputs = [outputs[conn] for conn in connections if conn is not None]
        gate_output = apply_gate(gate_type, *gate_inputs)
        outputs.append(gate_output)
    final_output = outputs[-1]
    simplified = simplify(final_output)
    return final_output, simplified


# Convert back to circuit
def sympy_to_list_circuit(expr: Symbol) -> Circuit:
    circuit = []
    gate_count = 0

    def process_expr(expr: Symbol) -> int:
        nonlocal gate_count
        if isinstance(expr, Symbol):
            return int(str(expr)[1:])  # Assuming input symbols are named x0, x1, x2, etc.
        if isinstance(expr, Not):
            input_id = process_expr(expr.args[0])
            gate_count += 1
            circuit.append(("NOT", input_id, None))
            return gate_count - 1
        if isinstance(expr, And | Or):
            gate_type = "AND" if isinstance(expr, And) else "OR"
            inputs = [process_expr(arg) for arg in expr.args]
            while len(inputs) > 2:
                gate_count += 1
                circuit.append((gate_type, inputs[0], inputs[1]))
                inputs = [gate_count - 1] + inputs[2:]
            gate_count += 1
            circuit.append((gate_type, inputs[0], inputs[1]))
            return gate_count - 1

    process_expr(expr)
    return circuit
