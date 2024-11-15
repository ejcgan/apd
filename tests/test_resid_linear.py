from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float

from spd.experiments.resid_linear.models import (
    ResidualLinearModel,
    ResidualLinearSPDFullRankModel,
    ResidualLinearSPDRankPenaltyModel,
)
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.run_spd import Config, ResidualLinearConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple ResidualLinear config that we can use in multiple tests
RESID_LINEAR_TASK_CONFIG = ResidualLinearConfig(
    task_name="residual_linear",
    init_scale=1.0,
    k=6,
    feature_probability=0.2,
    pretrained_model_path=Path(),  # We'll create this later
)


def test_resid_linear_rank_penalty_decomposition_happy_path() -> None:
    set_seed(0)
    n_features = 3
    d_embed = 2
    d_mlp = 3
    n_layers = 1
    label_coeffs = [1.5, 1.8, 1.1]

    device = "cpu"
    config = Config(
        seed=0,
        spd_type="rank_penalty",
        unit_norm_matrices=False,
        topk=1,
        batch_topk=True,
        param_match_coeff=1.0,
        topk_recon_coeff=1,
        topk_l2_coeff=1e-2,
        attribution_type="gradient",
        lr=1e-2,
        batch_size=8,
        steps=5,  # Run only a few steps for the test
        print_freq=2,
        image_freq=5,
        save_freq=None,
        lr_warmup_pct=0.01,
        lr_schedule="cosine",
        task_config=RESID_LINEAR_TASK_CONFIG,
    )

    assert isinstance(config.task_config, ResidualLinearConfig)
    # Create a pretrained model
    pretrained_model = ResidualLinearModel(
        n_features=n_features, d_embed=d_embed, d_mlp=d_mlp, n_layers=n_layers
    ).to(device)

    # Create the SPD model
    model = ResidualLinearSPDRankPenaltyModel(
        n_features=pretrained_model.n_features,
        d_embed=pretrained_model.d_embed,
        d_mlp=pretrained_model.d_mlp,
        n_layers=pretrained_model.n_layers,
        k=config.task_config.k,
        init_scale=config.task_config.init_scale,
    ).to(device)

    # Use the pretrained model's embedding matrix and don't train it further
    model.W_E.data[:, :] = pretrained_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(pretrained_model.n_layers):
        model.layers[i].linear1.bias.data[:] = (
            pretrained_model.layers[i].input_layer.bias.data.detach().clone()
        )
        model.layers[i].linear1.bias.requires_grad = False
        model.layers[i].linear2.bias.data[:] = (
            pretrained_model.layers[i].output_layer.bias.data.detach().clone()
        )
        model.layers[i].linear2.bias.requires_grad = False

    # Create dataset and dataloader
    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        label_coeffs=label_coeffs,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Set up param_map
    param_map = {}
    for i in range(pretrained_model.n_layers):
        param_map[f"layers.{i}.input_layer.weight"] = f"layers.{i}.input_layer.weight"
        param_map[f"layers.{i}.output_layer.weight"] = f"layers.{i}.output_layer.weight"

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out, _, _ = model(batch)
    initial_loss = torch.mean((labels - initial_out) ** 2).item()

    # Run optimize function
    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        param_map=param_map,
        plot_results_fn=None,
    )

    # Calculate final loss
    final_out, _, _ = model(batch)
    final_loss = torch.mean((labels - final_out) ** 2).item()

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"


def test_resid_linear_spd_equivalence() -> None:
    set_seed(0)
    n_features = 3
    d_embed = 2
    d_mlp = 3
    n_layers = 2
    k = 1  # Single subnetwork

    device = "cpu"

    target_model = ResidualLinearModel(
        n_features=n_features, d_embed=d_embed, d_mlp=d_mlp, n_layers=n_layers
    ).to(device)

    # Init all params to random values
    for param in target_model.parameters():
        param.data = torch.randn_like(param.data)

    # Create the SPD model with k=1
    spd_model = ResidualLinearSPDFullRankModel(
        n_features=n_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=k,
        init_scale=1.0,
    ).to(device)

    # Set up param_map for
    param_map = {}
    for i in range(n_layers):
        param_map[f"layers.{i}.input_layer.weight"] = f"layers.{i}.linear1.subnetwork_params"
        param_map[f"layers.{i}.input_layer.bias"] = f"layers.{i}.linear1.bias"
        param_map[f"layers.{i}.output_layer.weight"] = f"layers.{i}.linear2.subnetwork_params"
        param_map[f"layers.{i}.output_layer.bias"] = f"layers.{i}.linear2.bias"

    # Copy parameters from target model to SPD model
    for name, param in target_model.named_parameters():
        if name in param_map:
            spd_param = spd_model.get_parameter(param_map[name])
            if "weight" in name:
                # Need to transpose the weight matrix because it's in a Linear module which has
                # shape (mlp_dim, d_embed) but we need it to be (d_embed, mlp_dim)
                spd_param.data[0] = param.data.T
            else:
                spd_param.data[0] = param.data

    # Copy embedding matrix
    spd_model.W_E.data[:, :] = target_model.W_E.data.detach().clone()

    # Create a random input
    batch_size = 4
    input_data: Float[torch.Tensor, "batch n_features"] = torch.rand(
        batch_size, n_features, device=device
    )

    # Forward pass through both models
    target_output, _, target_post_acts = target_model(input_data)
    # Note that the target_post_acts should be the same as the "layer_acts" from the SPD model
    spd_output, spd_layer_acts, _ = spd_model(input_data)

    # Assert outputs are the same
    assert torch.allclose(target_output, spd_output, atol=1e-6), "Outputs do not match"

    # Assert activations are the same for all the matching activations that we have stored
    # We haven't stored the post/layer-activations for the biases in the SPD model, so we only
    # compare the activations for values that we have stored
    for layer_name, target_act in target_post_acts.items():
        if layer_name in spd_layer_acts:
            spd_act = spd_layer_acts[layer_name]
            assert torch.allclose(
                target_act, spd_act, atol=1e-6
            ), f"Activations do not match for layer {layer_name}"


def test_resid_linear_spd_rank_penalty_full_rank_equivalence() -> None:
    """Test that ResidualLinearSPDRankPenaltyModel output and internal acts match
    ResidualLinearSPDFullRankModel.

    We set the bias as the same for both models.
    We set the A and B matrices of ResidualLinearSPDRankPenaltyModel to be the SVD of the
    subnetwork params of ResidualLinearSPDFullRankModel. We could instead use an identity and the
    original subnetwork params, but this is a stronger test.
    """
    set_seed(0)

    batch_size = 4
    n_features = 3
    d_embed = 2
    d_mlp = 3
    n_layers = 2
    k = 2
    m = 1

    device = "cpu"

    # Create the full rank model
    full_rank_model = ResidualLinearSPDFullRankModel(
        n_features=n_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=k,
        init_scale=1.0,
    ).to(device)

    # Create random parameters
    for param in full_rank_model.parameters():
        nn.init.xavier_normal_(param)

    # Create the rank penalty model
    rank_penalty_model = ResidualLinearSPDRankPenaltyModel(
        n_features=n_features,
        d_embed=d_embed,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=k,
        init_scale=1.0,
        m=m,
    ).to(device)

    # Copy embedding matrix and biases
    rank_penalty_model.W_E.data = full_rank_model.W_E.data.clone()
    for i in range(n_layers):
        # Full rank has a bias for each k index, whereas rank penalty has no k index.
        # We thus sum over the k index in the full rank biases before copying to the rank penalty
        rank_penalty_model.layers[i].linear1.bias.data = (
            full_rank_model.layers[i].linear1.bias.data.sum(dim=0).clone()
        )
        rank_penalty_model.layers[i].linear2.bias.data = (
            full_rank_model.layers[i].linear2.bias.data.sum(dim=0).clone()
        )

    # For each layer and subnetwork, decompose the full rank parameters using SVD
    for i in range(n_layers):
        for j in range(k):
            # Input layer
            W_in = full_rank_model.layers[i].linear1.subnetwork_params[j]  # [d_embed, d_mlp]
            U_in, S_in, Vh_in = torch.linalg.svd(W_in, full_matrices=False)
            sqrt_S_in = torch.sqrt(S_in)
            rank_penalty_model.layers[i].linear1.A.data[j] = U_in * sqrt_S_in.view(1, -1)
            rank_penalty_model.layers[i].linear1.B.data[j] = sqrt_S_in.view(-1, 1) * Vh_in

            # Output layer
            W_out = full_rank_model.layers[i].linear2.subnetwork_params[j]  # [d_mlp, d_embed]
            U_out, S_out, Vh_out = torch.linalg.svd(W_out, full_matrices=False)
            sqrt_S_out = torch.sqrt(S_out)
            rank_penalty_model.layers[i].linear2.A.data[j] = U_out * sqrt_S_out.view(1, -1)
            rank_penalty_model.layers[i].linear2.B.data[j] = sqrt_S_out.view(-1, 1) * Vh_out

    # Create a random input
    input_data: Float[torch.Tensor, "batch n_features"] = torch.rand(
        batch_size, n_features, device=device
    )

    # Forward pass through both models
    full_rank_output, full_rank_layer_acts, _ = full_rank_model(input_data)
    rank_penalty_output, rank_penalty_layer_acts, _ = rank_penalty_model(input_data)

    # Assert outputs are the same
    assert torch.allclose(full_rank_output, rank_penalty_output, atol=1e-6), "Outputs do not match"

    # Assert activations are the same for all layers
    for layer_name in full_rank_layer_acts:
        full_rank_act = full_rank_layer_acts[layer_name]
        rank_penalty_act = rank_penalty_layer_acts[layer_name]
        assert torch.allclose(
            full_rank_act, rank_penalty_act, atol=1e-6
        ), f"Activations do not match for layer {layer_name}"
