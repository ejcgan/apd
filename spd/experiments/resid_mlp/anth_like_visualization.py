import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from train_resid_mlp import Config

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.utils import DatasetGeneratedDataLoader, set_seed


def plot_activations_single_features(
    config: Config,
    model: ResidualMLPModel,
):
    # Generate a batch of data that has one active feature (identity will do)
    half_batch_size = config.n_features
    batch_size = 2 * half_batch_size
    n_instances = config.n_instances
    pos_batch = torch.eye(half_batch_size, device=device) * 0.5
    neg_batch = -pos_batch
    batch = torch.cat([pos_batch, neg_batch], dim=0)
    # Copy for every instance
    batch = batch.unsqueeze(1).repeat(1, n_instances, 1)

    # Run the forward pass
    out, layer_pre_acts, layer_post_acts = model(batch)
    # Detach and convert to numpy
    out = out.detach().cpu().numpy()
    layer_pre_acts = {k: v.detach().cpu().numpy() for k, v in layer_pre_acts.items()}
    layer_post_acts = {k: v.detach().cpu().numpy() for k, v in layer_post_acts.items()}
    n_layers = model.n_layers

    # Visualize the activations of the output neurons as an array
    # for every instance, plot the activations of the output neurons in a grid (with colorbar)
    fig, axs = plt.subplots(2, n_instances, figsize=(n_instances * 10, 16))
    axs = np.array(axs)
    for i in range(n_instances):
        # split the activations into positive and negative
        pos_activations = out[0:half_batch_size, i, :]
        neg_activations = out[half_batch_size:batch_size, i, :]
        axs[0, i].imshow(pos_activations, aspect="auto")
        axs[1, i].imshow(neg_activations, aspect="auto")
        axs[0, i].set_title(f"Instance {i}")
        axs[0, i].set_xlabel("Output neuron")
        axs[0, i].set_ylabel("Input feature (positive)")
        axs[1, i].set_xlabel("Output neuron")
        axs[1, i].set_ylabel("Input feature (negative)")
    fig.suptitle(
        "Output activations when activating single input features (positively and negatively)"
    )
    plt.subplots_adjust(top=0.85)
    plt.colorbar(axs[0, 0].images[0], ax=axs, orientation="vertical", label="Activation")
    plt.savefig(plot_save_dir / "output_activations.png")
    plt.close()

    # Plot the activations of the post-activations of the layers, all in one plot
    # With layers along the x-axis and (instances, layers) along the y-axis
    fig, axs = plt.subplots(2 * n_layers, n_instances, figsize=(n_instances * 10, 16 * n_layers))
    axs = np.array(axs)
    for i in range(n_instances):
        for l in range(n_layers):
            # Get the positive and negative activations
            pos_activations = layer_pre_acts[f"layers.{l}.linear2"][
                0:half_batch_size, i, :
            ]  # Note I(Lee) think there is a naming issue in models.py regarding pre vs post acts
            neg_activations = layer_pre_acts[f"layers.{l}.linear2"][
                half_batch_size:batch_size, i, :
            ]
            axs[2 * l, i].imshow(pos_activations, aspect="auto")
            axs[2 * l + 1, i].imshow(neg_activations, aspect="auto")
            axs[2 * l, i].set_title(f"Instance {i}, Layer {l} (positive)")
            axs[2 * l + 1, i].set_title(f"Instance {i}, Layer {l} (negative)")
            axs[2 * l, i].set_xlabel("Neuron")
            axs[2 * l, i].set_ylabel("Input feature (positive)")
            axs[2 * l + 1, i].set_xlabel("Neuron")
            axs[2 * l + 1, i].set_ylabel("Input feature (negative)")
    fig.suptitle(
        "Hidden activations of MLP layers when activating single input features (positively and negatively)"
    )
    plt.subplots_adjust(top=0.85)
    plt.colorbar(axs[0, 0].images[0], ax=axs, orientation="vertical", label="Activation")
    plt.savefig(plot_save_dir / "layer_post_activations.png")
    plt.close()


if __name__ == "__main__":
    # Load model trained using train_resid_mlp.py
    # Set up device and seed
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)

    # Load the pretrained model from the path shown in file_context_0
    out_dir = Path(__file__).parent / "out"
    model_path = (
        out_dir
        # / "resid_mlp_identity_abs_n-instances10_n-features100_d-resid100_d-mlp100_n-layers1_seed0"
        / "resid_mlp_identity_abs_n-instances10_n-features100_d-resid100_d-mlp40_n-layers1_seed0"
    )

    plot_save_dir = model_path / "anth_like_visualization"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(model_path / "target_model_config.yaml") as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Load label coefficients
    with open(model_path / "label_coeffs.json") as f:
        label_coeffs = torch.tensor(json.load(f), device=device)

    # Initialize and load model
    model = ResidualMLPModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        d_embed=config.d_embed,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
        act_fn_name=config.act_fn_name,
        apply_output_act_fn=config.apply_output_act_fn,
        in_bias=config.in_bias,
        out_bias=config.out_bias,
    ).to(device)

    model.load_state_dict(torch.load(model_path / "target_model.pth"))
    model.eval()

    # Load the dataset
    dataset = ResidualMLPDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=config.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        data_generation_type=config.data_generation_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    plot_activations_single_features(config, model)


# %%
