# %% Imports

import einops
import matplotlib.pyplot as plt
import torch

from spd.experiments.piecewise.plotting import plot_matrix
from spd.experiments.resid_mlp.models import (
    ResidualMLPModel,
)
from spd.experiments.resid_mlp.plotting import (
    calculate_virtual_weights,
    plot_2d_snr,
    plot_individual_feature_response,
    relu_contribution_plot,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.train_resid_mlp import ResidMLPTrainConfig
from spd.types import ModelPath
from spd.utils import set_seed

# %% Load model and config
path: ModelPath = "wandb:spd-train-resid-mlp/runs/lkg96w24"
set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(path)
model = model.to(device)
train_config = ResidMLPTrainConfig(**train_config_dict)
dataset = ResidualMLPDataset(
    n_instances=train_config.resid_mlp_config.n_instances,
    n_features=train_config.resid_mlp_config.n_features,
    feature_probability=train_config.feature_probability,
    device=device,
    calc_labels=False,
    label_type=train_config.label_type,
    act_fn_name=train_config.resid_mlp_config.act_fn_name,
    label_fn_seed=train_config.label_fn_seed,
    label_coeffs=label_coeffs,
    data_generation_type=train_config.data_generation_type,
)
batch, labels = dataset.generate_batch(train_config.batch_size)

# %% Plot feature response with one active feature
fig = plot_individual_feature_response(
    lambda batch: model(batch)[0],
    model_config=train_config.resid_mlp_config,
    device=device,
    sweep=False,
)
fig = plot_individual_feature_response(
    lambda batch: model(batch)[0],
    model_config=train_config.resid_mlp_config,
    device=device,
    sweep=True,
)
plt.show()

# %% Show connection strength between ReLUs and features
virtual_weights = calculate_virtual_weights(model=model, device=device)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)  # type: ignore

relu_contribution_plot(
    ax1=ax1,
    ax2=ax2,
    all_diag_relu_conns=virtual_weights["diag_relu_conns"],
    model=model,
    device=device,
    instance_idx=0,
)
plt.show()

# %% Calculate S/N ratio for 1 and 2 active features.
fig = plot_2d_snr(model, device)
plt.show()

# %% Plot virtual weights

fig = plt.figure(constrained_layout=True, figsize=(20, 20))
gs = fig.add_gridspec(ncols=2, nrows=3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1:, :])
virtual_weights = calculate_virtual_weights(model=model, device=device)
instance_idx = 0
in_conns = virtual_weights["in_conns"][instance_idx].cpu().detach()
out_conns = virtual_weights["out_conns"][instance_idx].cpu().detach()
W_E_W_U = einops.einsum(
    virtual_weights["W_E"][instance_idx],
    virtual_weights["W_U"][instance_idx],
    "n_features1 d_embed, d_embed n_features2 -> n_features1 n_features2",
)
plot_matrix(
    ax1,
    in_conns.T,
    "Virtual input weights $(W_E W_{in})^T$",
    "Features",
    "Neurons",
    colorbar_format="%.2f",
)
plot_matrix(
    ax2,
    out_conns,
    "Virtual output weights $W_{out} W_U$",
    "Features",
    "Neurons",
    colorbar_format="%.2f",
)
ax2.xaxis.set_label_position("top")
plot_matrix(
    ax3,
    W_E_W_U,
    "Virtual weights $W_E W_U$",
    "Features",
    "Features",
    colorbar_format="%.2f",
)
plt.show()

# %%
