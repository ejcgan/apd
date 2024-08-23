"""Function to hand-code a boolean circuit as a neural network"""

import torch

from spd.experiments.bool_circuits.bool_circuit_utils import (
    AndOperation,
    BooleanOperation,
    NotOperation,
    OrOperation,
    create_circuit_str,
    evaluate_circuit,
    make_detailed_circuit,
)
from spd.experiments.bool_circuits.models import BoolCircuitTransformer

# %%
num_inputs = 10
circuit: list[BooleanOperation] = [
    OrOperation(8, 4),
    NotOperation(3, None),
    AndOperation(10, 9),
    OrOperation(9, 4),
    OrOperation(7, 10),
    NotOperation(11, None),
    OrOperation(2, 10),
    NotOperation(3, None),
    OrOperation(10, 6),
    AndOperation(0, 8),
    AndOperation(7, 11),
    AndOperation(10, 13),
    AndOperation(3, 4),
    NotOperation(7, None),
    AndOperation(18, 20),
]
print("Original circuit:")
for i, op in enumerate(circuit):
    print(f"{i}: {op.name} {op.input_idx1} {op.input_idx2}")

final_output = create_circuit_str(circuit, num_inputs)
print("expression:", final_output)

detailed_circuit = make_detailed_circuit(circuit, num_inputs)
print("Detailed circuit:")
for i, op in enumerate(detailed_circuit):
    print(
        f"{i}: {op.name} {op.input_idx1} {op.input_idx2} -> {op.out_idx} @ layer {op.min_layer_needed}"
    )


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
    label = evaluate_circuit([bool(x) for x in random_bools[0]], circuit)
    assert net == label, f"{net=}, {label=}"
# %%
