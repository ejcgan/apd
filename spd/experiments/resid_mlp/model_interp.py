# %% Imports
import matplotlib.pyplot as plt
import torch

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.plotting import (
    plot_2d_snr,
    plot_individual_feature_response,
    plot_virtual_weights,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.experiments.resid_mlp.train_resid_mlp import ResidMLPTrainConfig
from spd.types import ModelPath
from spd.utils import set_seed

# %% Load model and config
# path = (
#     REPO_ROOT
#     / "spd/experiments/resid_mlp/out/resid_mlp_identity_act_plus_resid_n-instances2_n-features100_d-resid1000_d-mlp50_n-layers1_seed0/target_model.pth"
# )
path: ModelPath = "wandb:spd-train-resid-mlp/runs/njgvwytn"

set_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, task_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(path)
model = model.to(device)
task_config = ResidMLPTrainConfig(**task_config_dict)
dataset = ResidualMLPDataset(
    n_instances=task_config.resid_mlp_config.n_instances,
    n_features=task_config.resid_mlp_config.n_features,
    feature_probability=task_config.feature_probability,
    device=device,
    calc_labels=False,
    label_type=task_config.label_type,
    act_fn_name=task_config.resid_mlp_config.act_fn_name,
    label_fn_seed=task_config.label_fn_seed,
    label_coeffs=label_coeffs,
    data_generation_type=task_config.data_generation_type,
)
batch, labels = dataset.generate_batch(task_config.batch_size)

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

# %% Calculate S/N ratio for 1 and 2 active features.
fig = plot_2d_snr(model, device)
plt.show()

# %% Plot virtual weights
fig = plot_virtual_weights(model, device)
plt.show()

# %%
