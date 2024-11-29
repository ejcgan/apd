# %% Imports
import matplotlib.pyplot as plt
import torch

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.plotting import (
    plot_2d_snr,
    plot_individual_feature_response,
    plot_virtual_weights,
    relu_contribution_plot,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.utils import REPO_ROOT, set_seed

# %% Load model and config
model = "resid_mlp_identity_act_plus_resid_n-instances20_n-features100_d-resid1000_d-mlp10_n-layers5_seed0"
path = REPO_ROOT / "spd/experiments/resid_mlp/out" / model / "target_model.pth"

set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, task_config, label_coeffs = ResidualMLPModel.from_pretrained(path)
model = model.to(device)
dataset = ResidualMLPDataset(
    n_instances=task_config["n_instances"],
    n_features=task_config["n_features"],
    feature_probability=task_config["feature_probability"],
    device=device,
    label_type=task_config["label_type"],
    act_fn_name=task_config["act_fn_name"],
    label_fn_seed=task_config["label_fn_seed"],
    label_coeffs=label_coeffs,
    data_generation_type=task_config["data_generation_type"],
)
batch, labels = dataset.generate_batch(task_config["batch_size"])

# %% Plot feature response with one active feature
fig = plot_individual_feature_response(
    model,
    device,
    task_config,
    sweep=False,
)
fig = plot_individual_feature_response(
    model,
    device,
    task_config,
    sweep=True,
)
plt.show()

# %% Show connection strength between ReLUs and features
fig = relu_contribution_plot(model, device)
plt.show()

# %% Calculate S/N ratio for 1 and 2 active features.
fig = plot_2d_snr(model, device)
plt.show()


# %% Plot virtual weights
fig = plot_virtual_weights(model, device)
plt.show()

# %%
