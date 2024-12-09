# %% Imports


import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLPModel, ResidualMLPSPDRankPenaltyModel
from spd.experiments.resid_mlp.plotting import (
    analyze_per_feature_performance,
    plot_individual_feature_response,
    plot_spd_relu_contribution,
    plot_virtual_weights_target_spd,
    spd_calculate_virtual_weights,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import ResidualMLPTaskConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %% Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
set_seed(0)  # You can change this seed if needed
wandb_path = "wandb:spd-resid-mlp/runs/50ngtqb5"
# Load the pretrained SPD model
model, config, label_coeffs = ResidualMLPSPDRankPenaltyModel.from_pretrained(wandb_path)
assert isinstance(config.task_config, ResidualMLPTaskConfig)
# Path must be local
target_model, target_model_train_config_dict, target_label_coeffs = (
    ResidualMLPModel.from_pretrained(config.task_config.pretrained_model_path)
)
model = model.to(device)
label_coeffs = label_coeffs.to(device)
target_model = target_model.to(device)
target_label_coeffs = target_label_coeffs.to(device)
assert torch.allclose(target_label_coeffs, torch.tensor(label_coeffs))
dataset = ResidualMLPDataset(
    n_instances=model.config.n_instances,
    n_features=model.config.n_features,
    feature_probability=config.task_config.feature_probability,
    device=device,
    calc_labels=False,  # Our labels will be the output of the target model
    data_generation_type=config.task_config.data_generation_type,
)
batch, labels = dataset.generate_batch(config.batch_size)
batch = batch.to(device)
labels = labels.to(device)
# Print some basic information about the model
print(f"Number of features: {model.config.n_features}")
print(f"Embedding dimension: {model.config.d_embed}")
print(f"MLP dimension: {model.config.d_mlp}")
print(f"Number of layers: {model.config.n_layers}")
print(f"Number of subnetworks (k): {model.config.k}")

target_model_output, _, _ = target_model(batch)

assert config.topk is not None
spd_outputs = run_spd_forward_pass(
    spd_model=model,
    target_model=target_model,
    input_array=batch,
    attribution_type=config.attribution_type,
    batch_topk=config.batch_topk,
    topk=config.topk,
    distil_from_target=config.distil_from_target,
)
topk_recon_loss = calc_recon_mse(
    spd_outputs.spd_topk_model_output, target_model_output, has_instance_dim=True
)
print(f"Topk recon loss: {np.array(topk_recon_loss.detach().cpu())}")

# Print param shapes for model
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


# %% Feature-relu contribution plots

fig1, fig2 = plot_spd_relu_contribution(model, target_model, device, k_plot_limit=3)
fig1.suptitle("How much does each ReLU contribute to each feature?")
fig2.suptitle("How much does each feature route through each ReLU?")


# %% Individual feature response
def spd_model_fn(batch: Float[Tensor, "batch n_instances"]):
    assert config.topk is not None
    return run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        attribution_type=config.attribution_type,
        batch_topk=config.batch_topk,
        topk=config.topk,
        distil_from_target=config.distil_from_target,
    ).spd_topk_model_output


def target_model_fn(batch: Float[Tensor, "batch n_instances"]):
    return target_model(batch)[0]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), constrained_layout=True)
axes = np.atleast_2d(axes)  # type: ignore
plot_individual_feature_response(
    model_fn=target_model_fn,
    device=device,
    model_config=model.config,
    ax=axes[0, 0],
)
plot_individual_feature_response(
    model_fn=target_model_fn,
    device=device,
    model_config=model.config,
    sweep=True,
    ax=axes[1, 0],
)

plot_individual_feature_response(
    model_fn=spd_model_fn,
    device=device,
    model_config=model.config,
    ax=axes[0, 1],
)
plot_individual_feature_response(
    model_fn=spd_model_fn,
    device=device,
    model_config=model.config,
    sweep=True,
    ax=axes[1, 1],
)
axes[0, 0].set_ylabel(axes[0, 0].get_title())
axes[1, 0].set_ylabel(axes[1, 0].get_title())
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")
axes[0, 0].set_title("Target model")
axes[0, 1].set_title("SPD model")
axes[1, 0].set_title("")
axes[1, 1].set_title("")
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
fig.show()

# %% Per-feature performance
fig, ax = plt.subplots(figsize=(15, 5))
sorted_indices = analyze_per_feature_performance(
    model_fn=spd_model_fn,
    model_config=model.config,
    ax=ax,
    label="SPD",
    device=device,
    sorted_indices=None,
    zorder=1,
)
analyze_per_feature_performance(
    model_fn=target_model_fn,
    model_config=target_model.config,
    ax=ax,
    label="Target",
    device=device,
    sorted_indices=sorted_indices,
    zorder=0,
)
ax.legend()
fig.show()


# %% Virtual weights
fig = plot_virtual_weights_target_spd(target_model, model, device)
fig.show()

# %% Analysis of one feature / subnetwork, picking feature 1 because it looks sketch.

# Subnet combinations relevant for feature 1
virtual_weights = spd_calculate_virtual_weights(model, device)
in_conns: Float[Tensor, "k1 n_features1 d_mlp"] = virtual_weights["in_conns"][0]
out_conns: Float[Tensor, "k2 d_mlp n_features2"] = virtual_weights["out_conns"][0]
relu_conns_sum: Float[Tensor, "k1 k2 f1 f2"] = einops.einsum(
    in_conns, out_conns, "k1 f1 d_mlp, k2 d_mlp f2 -> k1 k2 f1 f2"
)
plt.matshow(relu_conns_sum[:, :, 1, 1].detach().cpu())
plt.title("Subnet combinations relevant for feature 1")
plt.show()

# Per-neuron contribution to feature 1
relu_conns: Float[Tensor, "k1 k2 f1 f2"] = einops.einsum(
    in_conns, out_conns, "k1 f1 d_mlp, k2 d_mlp f2 -> k1 k2 f1 f2 d_mlp"
)
plt.plot(relu_conns[1, 1, 1, 1, :].detach().cpu(), label="Subnet 1 of W_in and W_out")
plt.plot(
    relu_conns[:, :, 1, 1, :].sum(dim=(0, 1)).detach().cpu(),
    label="All subnets (i,j) of W_in and W_out",
)
plt.plot(
    relu_conns[:, :, 1, 1, :].sum(dim=(0, 1)).detach().cpu()
    - relu_conns[1, 1, 1, 1, :].detach().cpu(),
    label="All subnets (i,j) != (1,1) of W_in and W_out",
)
plt.title("Per-neuron contribution to feature 1")
plt.xlabel("Neuron")
plt.ylabel("Weight")
plt.legend()
plt.show()

# Which subnets contain the neuron-45 contribution to feature 1?
plt.matshow(relu_conns[:, :, 1, 1, 45].detach().cpu())
plt.title("Which subnets contain the neuron-45 contribution to feature 1?")
print("Seems to be the diagonal k1=95, k2=95 term", relu_conns[:, :, 1, 1, 45].argmax())

# %%
