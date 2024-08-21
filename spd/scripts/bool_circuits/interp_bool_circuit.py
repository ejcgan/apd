# %%
import json
from pathlib import Path

import torch

from spd.log import logger
from spd.scripts.bool_circuits.bool_circuit_utils import (
    create_circuit_str,
    create_truth_table,
    make_detailed_circuit,
    plot_circuit,
)
from spd.scripts.bool_circuits.models import BoolCircuitTransformer
from spd.scripts.bool_circuits.train_bool_circuit import (
    Config,
    evaluate_model,
    get_circuit,
    get_train_test_dataloaders,
)
from spd.types import RootPath

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Load model, config, circuit, truth table, dataloaders, and evaluate model

out_dir: RootPath = Path(__file__).parent / "out/inp10-op20-hid8-lay1-circseed1-seed0"

with open(out_dir / "config.json") as f:
    config = Config(**json.load(f))
logger.info(f"Config loaded from {out_dir / 'config.json'}")

trained_model = BoolCircuitTransformer(
    n_inputs=config.n_inputs,
    d_embed=config.d_embed,
    d_mlp=config.d_embed,
    n_layers=config.n_layers,
).to(device)
trained_model.load_state_dict(
    torch.load(out_dir / "model.pt", weights_only=True, map_location="cpu")
)
logger.info(f"Model loaded from {out_dir / 'model.pt'}")

circuit = get_circuit(config)
circuit = make_detailed_circuit(circuit, config.n_inputs)
plot_circuit(circuit, config.n_inputs, show_out_idx=True)
logger.info(f"Circuit: n_inputs={config.n_inputs} - {circuit}")
logger.info(f"Circuit string: {create_circuit_str(circuit, config.n_inputs)}")

handcoded_model = BoolCircuitTransformer(
    n_inputs=config.n_inputs,
    d_embed=30,  # exact minimum for my handcode of that circuit
    d_mlp=7,  # exact minimum for my handcode of that circuit
    n_layers=7,  # exact minimum for my handcode of that circuit
).to(device)
handcoded_model.init_handcoded(circuit)
logger.info("Handcoded model initialized with circuit")


truth_table = create_truth_table(config.n_inputs, circuit)
logger.info(f"Truth table:\n{truth_table}")

train_dataloader, eval_dataloader = get_train_test_dataloaders(config, circuit)
eval_loss = evaluate_model(trained_model, eval_dataloader, device)
logger.info(f"Evaluation loss: {eval_loss}")
assert eval_loss < 1e-5, f"Unusual evaluation loss of {eval_loss:.3e}"

eval_loss_handcoded = evaluate_model(
    handcoded_model, eval_dataloader, device, output_is_logit=False
)
logger.info(f"Handcoded model evaluation loss: {eval_loss_handcoded}")


# %%

for inputs, labels in train_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    print(f"{inputs.shape=}, {inputs}")
    preds = trained_model(inputs)
    probabilities = torch.sigmoid(preds)
    print(f"{preds.shape=}, {preds:}")
    print(f"{probabilities.shape=}, {probabilities}")
    print(f"{labels.shape=}, {labels}")
    break

# %%


# %%
