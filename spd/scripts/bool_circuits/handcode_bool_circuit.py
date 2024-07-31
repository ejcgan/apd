"""Function to hand-code a boolean circuit as a neural network"""

import torch

from spd.scripts.bool_circuits.bool_circuit_model import BoolCircuitTransformer
from spd.scripts.bool_circuits.bool_circuit_utils import (
    BooleanOperation,
    create_circuit_str,
    evaluate_circuit,
    make_detailed_circuit,
)

# %%

num_inputs, circuit = (
    10,
    [
        BooleanOperation("OR", 8, 4),
        BooleanOperation("NOT", 3, None),
        BooleanOperation("AND", 10, 9),
        BooleanOperation("OR", 9, 4),
        BooleanOperation("OR", 7, 10),
        BooleanOperation("NOT", 11, None),
        BooleanOperation("OR", 2, 10),
        BooleanOperation("NOT", 3, None),
        BooleanOperation("OR", 10, 6),
        BooleanOperation("AND", 0, 8),
        BooleanOperation("AND", 7, 11),
        BooleanOperation("AND", 10, 13),
        BooleanOperation("AND", 3, 4),
        BooleanOperation("NOT", 7, None),
        BooleanOperation("AND", 18, 20),
    ],
)
print("Original circuit:")
for i, op in enumerate(circuit):
    print(f"{i}: {op.op_name} {op.arg1} {op.arg2}")

final_output = create_circuit_str(circuit, num_inputs)
print("expression:", final_output)

detailed_circuit = make_detailed_circuit(circuit, num_inputs)
print("Detailed circuit:")
for i, op in enumerate(detailed_circuit):
    print(f"{i}: {op.op_name} {op.arg1} {op.arg2} -> {op.out_idx} @ layer {op.min_layer_needed}")


d_hidden = 40
n_layers = 6
network = BoolCircuitTransformer(
    n_inputs=num_inputs, d_embed=d_hidden, d_mlp=d_hidden, n_layers=n_layers
)
network.init_handcoded(circuit)


# %%
for _ in range(1000):
    random_bools = torch.randint(0, 2, (1, num_inputs))
    net = network(random_bools.float()).item()
    random_bools_int: list[int] = [int(x) for x in random_bools[0]]
    label = evaluate_circuit(random_bools_int, circuit)
    assert net == label, f"{net=}, {label=}"
# %%
