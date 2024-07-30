"""Function to hand-code a boolean circuit as a neural network"""

import torch

from spd.scripts.bool_circuits.boolean_circuit import BoolCircuitModel
from spd.scripts.bool_circuits.circuit_utils import (
    create_circuit_str,
    evaluate_circuit,
    make_detailed_circuit,
)

# %% Use sympy to simplify the circuit

num_inputs, circuit = (
    10,
    [
        ("OR", 8, 4),
        ("NOT", 3, None),
        ("AND", 10, 9),
        ("OR", 9, 4),
        ("OR", 7, 10),
        ("NOT", 11, None),
        ("OR", 2, 10),
        ("NOT", 3, None),
        ("OR", 10, 6),
        ("AND", 0, 8),
        ("AND", 7, 11),
        ("AND", 10, 13),
        ("AND", 3, 4),
        ("NOT", 7, None),
        ("AND", 18, 20),
    ],
)
print("Original circuit:")
for i, (gate, arg1, arg2) in enumerate(circuit):
    print(f"{i}: {gate} {arg1} {arg2}")

final_output = create_circuit_str(circuit, num_inputs)
print("expression:", final_output)

detailed_circuit = make_detailed_circuit(circuit, num_inputs)
print("Detailed circuit:")
for i, (gate, arg1, arg2, out_idx, min_layer_needed) in enumerate(detailed_circuit):
    print(f"{i}: {gate} {arg1} {arg2} -> {out_idx} @ layer {min_layer_needed}")


d_hidden = 40
n_layers = 6
network = BoolCircuitModel(n_inputs=num_inputs, d_hidden=d_hidden, n_layers=n_layers)
network.hand_coded_implementation(circuit)


# %%
for _ in range(1000):
    random_bools = torch.randint(0, 2, (1, num_inputs))
    net = network(random_bools.float()).item()
    label = evaluate_circuit(list(random_bools[0]), circuit)
    assert net == label, f"{net=}, {label=}"
# %%
