from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from train_resid_mlp import ResidMLPTrainConfig

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.types import ModelPath
from spd.utils import DatasetGeneratedDataLoader, set_seed


def plot_activations_single_features(model: ResidualMLPModel):
    # Generate a batch of data that has one active feature (identity will do)
    half_batch_size = model.config.n_features
    batch_size = 2 * half_batch_size
    n_instances = model.config.n_instances
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
    n_layers = model.config.n_layers

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
    print(f"Saved to {plot_save_dir / 'output_activations.png'}")
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
    print(f"Saved to {plot_save_dir / 'layer_post_activations.png'}")
    plt.close()


if __name__ == "__main__":
    # Load model trained using train_resid_mlp.py
    # Set up device and seed

    # Use either local path or wandb path
    # model_path: ModelPath = Path(
    #     "spd/experiments/resid_mlp/out/resid_mlp_identity_abs_n-instances5_n-features10_d-resid5_d-mlp5_n-layers1_seed0/resid_mlp.pth"
    # )
    model_path: ModelPath = "wandb:spd-train-resid-mlp/runs/s7h0jsco"

    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)

    if isinstance(model_path, str):
        # wandb path
        run_id = model_path.split("/")[-1]
        plot_save_dir = Path(__file__).parent / "out" / run_id / "anth_like_visualization"
    else:
        # local path
        plot_save_dir = model_path.parent / "anth_like_visualization"
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    model, train_config_dict, label_coeffs = ResidualMLPModel.from_pretrained(model_path)
    model.eval()

    config = ResidMLPTrainConfig(**train_config_dict)

    # Load the dataset
    dataset = ResidualMLPDataset(
        n_instances=config.resid_mlp_config.n_instances,
        n_features=config.resid_mlp_config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=config.resid_mlp_config.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        label_coeffs=None,  # TODO Is this intended?
        data_generation_type=config.data_generation_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    plot_activations_single_features(model)


# %%
