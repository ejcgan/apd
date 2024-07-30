from sympy import And, Not, Or, Symbol, simplify, symbols


def evaluate_circuit(inputs: list[int], circuit: list[tuple[str, int, int]]) -> int:
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


def make_detailled_circuit(
    base_circuit: list[tuple[str, int, int]], n_inputs: int
) -> list[tuple[str, int, int, int, int]]:
    """Adds (i) the output index and (ii) the minimum layer to implement the gate, to the circuit"""
    # New format for the circuit: (gate, arg1, arg2, out_idx, min_layer_needed)
    circuit: list[tuple[str, int, int, int, int]] = []

    # Assign an output index to each gate
    for i, (gate, arg1, arg2) in enumerate(base_circuit):
        circuit.append([gate, arg1, arg2, n_inputs + i, None])

    # Find the minimum layer needed for each gate
    fully_specificed = False
    while not fully_specificed:
        for i, (gate, arg1, arg2, out_idx, min_layer_needed) in enumerate(circuit):  # noqa: B007
            # Continue if we have already determined the layer for this gate
            if min_layer_needed is not None:
                continue
            # Determine the layer at which the inputs are available
            layer1 = 0 if arg1 < n_inputs else circuit[arg1 - n_inputs][4]
            layer2 = 0 if gate == "NOT" or arg2 < n_inputs else circuit[arg2 - n_inputs][4]
            # Continue if we haven't determined the layer for one of the inputs *yet*
            if layer1 is None or layer2 is None:
                continue
            # Set the layer at which this gate's output is available
            circuit[i][4] = max(layer1, layer2) + 1
        # Check if the algorithm has convered
        fully_specificed = all([x[4] is not None for x in circuit])

    return circuit


def list_circuit_to_sympy(circuit, num_inputs):
    inputs = symbols(f"x:{num_inputs}")

    def apply_gate(gate_type, *args):
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
def sympy_to_list_circuit(expr):
    circuit = []
    gate_inputs = {}
    gate_count = 0

    def process_expr(expr):
        nonlocal gate_count
        if isinstance(expr, Symbol):
            return int(str(expr)[1:])  # Assuming input symbols are named x0, x1, x2, etc.
        if isinstance(expr, Not):
            input_id = process_expr(expr.args[0])
            gate_count += 1
            circuit.append(("NOT", input_id, None))
            return gate_count - 1
        if isinstance(expr, And) or isinstance(expr, Or):
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
