from pathlib import Path

import torch
from jaxtyping import Float

from spd.experiments.resid_mlp.models import (
    ResidualMLPConfig,
    ResidualMLPModel,
    ResidualMLPSPDRankPenaltyConfig,
    ResidualMLPSPDRankPenaltyModel,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.run_spd import Config, ResidualMLPTaskConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple ResidualMLP config that we can use in multiple tests
RESID_MLP_TASK_CONFIG = ResidualMLPTaskConfig(
    task_name="residual_mlp",
    k=3,
    feature_probability=0.333,
    init_scale=1.0,
    data_generation_type="at_least_zero_active",
    pretrained_model_path=Path(),  # We'll create this later
)


def test_resid_mlp_rank_penalty_decomposition_happy_path() -> None:
    # Just noting that this test will only work on 98/100 seeds. So it's possible that future
    # changes will break this test.
    set_seed(0)
    resid_mlp_config = ResidualMLPConfig(
        n_instances=2,
        n_features=3,
        d_embed=2,
        d_mlp=3,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=True,
        out_bias=True,
    )

    device = "cpu"
    config = Config(
        seed=0,
        spd_type="rank_penalty",
        topk=1,
        batch_topk=True,
        param_match_coeff=1.0,
        topk_recon_coeff=1,
        topk_l2_coeff=1e-2,
        attribution_type="gradient",
        lr=1e-3,
        batch_size=32,
        steps=10,  # Run only a few steps for the test
        print_freq=2,
        image_freq=5,
        save_freq=None,
        lr_warmup_pct=0.01,
        lr_schedule="cosine",
        task_config=RESID_MLP_TASK_CONFIG,
    )

    assert isinstance(config.task_config, ResidualMLPTaskConfig)
    # Create a pretrained model
    target_model = ResidualMLPModel(config=resid_mlp_config).to(device)

    # Create the SPD model
    spd_config = ResidualMLPSPDRankPenaltyConfig(
        **resid_mlp_config.model_dump(), k=config.task_config.k
    )
    model = ResidualMLPSPDRankPenaltyModel(config=spd_config).to(device)

    # Use the pretrained model's embedding matrices and don't train them further
    model.W_E.data[:, :] = target_model.W_E.data.detach().clone()
    model.W_E.requires_grad = False
    model.W_U.data[:, :] = target_model.W_U.data.detach().clone()
    model.W_U.requires_grad = False

    # Copy the biases from the target model to the SPD model and set requires_grad to False
    for i in range(resid_mlp_config.n_layers):
        if resid_mlp_config.in_bias:
            model.layers[i].linear1.bias.data[:, :] = (
                target_model.layers[i].bias1.data.detach().clone()
            )
            model.layers[i].linear1.bias.requires_grad = False
        if resid_mlp_config.out_bias:
            model.layers[i].linear2.bias.data[:, :] = (
                target_model.layers[i].bias2.data.detach().clone()
            )
            model.layers[i].linear2.bias.requires_grad = False

    # Create dataset and dataloader
    dataset = ResidualMLPDataset(
        n_instances=model.n_instances,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        calc_labels=False,
        label_type=None,
        act_fn_name=None,
        label_fn_seed=None,
        label_coeffs=None,
        data_generation_type="at_least_zero_active",
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Set up param_map
    param_map = {}
    for i in range(resid_mlp_config.n_layers):
        param_map[f"layers.{i}.linear1"] = f"layers.{i}.linear1"
        param_map[f"layers.{i}.linear2"] = f"layers.{i}.linear2"

    # Calculate initial loss
    with torch.inference_mode():
        batch, _ = next(iter(dataloader))
        initial_out, _, _ = model(batch)
        labels, _, _ = target_model(batch)
        initial_loss = torch.mean((labels - initial_out) ** 2).item()

    # Run optimize function
    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=target_model,
        param_map=param_map,
        plot_results_fn=None,
    )

    # Calculate final loss
    with torch.inference_mode():
        final_out, _, _ = model(batch)
        final_loss = torch.mean((labels - final_out) ** 2).item()

    print(f"Final loss: {final_loss}, initial loss: {initial_loss}")
    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"

    # Show that W_E is still the same as the target model's W_E
    assert torch.allclose(model.W_E, target_model.W_E, atol=1e-6)


def test_resid_mlp_rank_penalty_equivalent_to_raw_model() -> None:
    device = "cpu"
    set_seed(0)
    resid_mlp_config = ResidualMLPConfig(
        n_instances=2,
        n_features=3,
        d_embed=2,
        d_mlp=3,
        n_layers=2,
        act_fn_name="relu",
        apply_output_act_fn=False,
        in_bias=True,
        out_bias=True,
    )
    k = 2  # Single subnetwork

    target_model = ResidualMLPModel(config=resid_mlp_config).to(device)

    # Create the SPD model with k=1
    resid_mlp_spd_config = ResidualMLPSPDRankPenaltyConfig(**resid_mlp_config.model_dump(), k=k)
    spd_model = ResidualMLPSPDRankPenaltyModel(config=resid_mlp_spd_config).to(device)

    # Init all params to random values
    for param in spd_model.parameters():
        param.data = torch.randn_like(param.data)

    # Copy weight matrices from SPD model to target model
    spd_params = spd_model.all_subnetwork_params_summed()
    for name, param in target_model.named_parameters():
        if name in spd_params:
            param.data[:, :, :] = spd_params[name].data

    # Also copy the embeddings and biases
    target_model.W_E.data[:, :, :] = spd_model.W_E.data
    target_model.W_U.data[:, :, :] = spd_model.W_U.data
    for i in range(resid_mlp_config.n_layers):
        target_model.layers[i].bias1.data[:, :] = spd_model.layers[i].linear1.bias.data
        target_model.layers[i].bias2.data[:, :] = spd_model.layers[i].linear2.bias.data

    # Create a random input
    batch_size = 4
    input_data: Float[torch.Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, resid_mlp_config.n_instances, resid_mlp_config.n_features, device=device
    )

    with torch.inference_mode():
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
