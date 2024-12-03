from pathlib import Path

import pytest
import torch
import torch.nn as nn
from jaxtyping import Float

from spd.experiments.tms.models import (
    TMSModel,
    TMSSPDFullRankModel,
    TMSSPDRankPenaltyModel,
)
from spd.experiments.tms.train_tms import TMSTrainConfig, get_model_and_dataloader, train
from spd.run_spd import Config, TMSConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, SparseFeatureDataset, set_seed

# Create a simple TMS config that we can use in multiple tests
TMS_TASK_CONFIG = TMSConfig(
    task_name="tms",
    n_features=5,
    n_hidden=2,
    n_instances=2,
    k=5,
    feature_probability=0.5,
    train_bias=False,
    bias_val=0.0,
    pretrained_model_path=Path(""),  # We'll create this later
)


def tms_spd_rank_penalty_happy_path(config: Config, n_hidden_layers: int = 0):
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, TMSConfig)

    model = TMSSPDRankPenaltyModel(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        n_hidden=config.task_config.n_hidden,
        n_hidden_layers=n_hidden_layers,
        k=config.task_config.k,
        bias_val=config.task_config.bias_val,
        device=device,
    )
    # For our pretrained model, just use a randomly initialized TMS model
    pretrained_model = TMSModel(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        n_hidden=config.task_config.n_hidden,
        n_hidden_layers=n_hidden_layers,
        device=device,
    )
    # Randomly initialize the bias for the pretrained model
    pretrained_model.b_final.data = torch.randn_like(pretrained_model.b_final.data)
    # Manually set the bias for the SPD model from the bias in the pretrained model
    model.b_final.data[:] = pretrained_model.b_final.data.clone()

    if not config.task_config.train_bias:
        model.b_final.requires_grad = False

    dataset = SparseFeatureDataset(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        data_generation_type=config.task_config.data_generation_type,
        value_range=(0.0, 1.0),
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    # Pick an arbitrary parameter to check that it changes
    initial_param = model.A.clone().detach()

    param_map = {"W": "W", "W_T": "W_T"}
    if model.hidden_layers is not None:
        for i in range(len(model.hidden_layers)):
            param_map[f"hidden_{i}"] = f"hidden_{i}"

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

    assert not torch.allclose(
        initial_param, model.A
    ), "Model A matrix should have changed after optimization"


def test_tms_batch_topk_no_schatten():
    config = Config(
        spd_type="rank_penalty",
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        schatten_pnorm=None,
        schatten_coeff=None,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_rank_penalty_happy_path(config)


@pytest.mark.parametrize("n_hidden_layers", [0, 2])
def test_tms_batch_topk_and_schatten(n_hidden_layers: int):
    config = Config(
        spd_type="rank_penalty",
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        schatten_pnorm=0.9,
        schatten_coeff=1e-1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_rank_penalty_happy_path(config, n_hidden_layers)


def test_tms_topk_and_l2():
    config = Config(
        spd_type="rank_penalty",
        topk=2,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        schatten_pnorm=0.9,
        schatten_coeff=1e-1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_rank_penalty_happy_path(config)


def test_tms_lp():
    config = Config(
        spd_type="rank_penalty",
        topk=None,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_rank_penalty_happy_path(config)


@pytest.mark.parametrize("n_hidden_layers", [0, 2])
def test_tms_topk_and_lp(n_hidden_layers: int):
    config = Config(
        spd_type="rank_penalty",
        topk=2,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        pnorm=0.9,
        topk_recon_coeff=1,
        lp_sparsity_coeff=1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_rank_penalty_happy_path(config, n_hidden_layers)


def test_train_tms_happy_path():
    device = "cpu"
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        n_features=3,
        n_hidden=2,
        n_instances=2,
        n_hidden_layers=0,
        feature_probability=0.1,
        batch_size=32,
        steps=5,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out, _, _ = model(batch)
    initial_loss = torch.mean((labels.abs() - initial_out) ** 2)

    train(model, dataloader, steps=config.steps, print_freq=1000)

    # Calculate final loss
    final_out, _, _ = model(batch)
    final_loss = torch.mean((labels.abs() - final_out) ** 2)

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Final loss ({final_loss:.2e}) is not lower than initial loss ({initial_loss:.2e})"


def test_tms_train_fixed_identity():
    """Check that hidden layer is identity before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        n_features=3,
        n_hidden=2,
        n_instances=2,
        n_hidden_layers=2,
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=True,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    eye = torch.eye(config.n_hidden, device=device).expand(config.n_instances, -1, -1)

    assert model.hidden_layers is not None
    # Assert that this is an identity matrix
    initial_hidden = model.hidden_layers[0].data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(model, dataloader, steps=config.steps, print_freq=1000)

    # Assert that the hidden layers remains identity
    assert torch.allclose(model.hidden_layers[0].data, eye), "Hidden layer changed"


def test_tms_train_fixed_random():
    """Check that hidden layer is random before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        n_features=3,
        n_hidden=2,
        n_instances=2,
        n_hidden_layers=2,
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=False,
        fixed_random_hidden_layers=True,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    assert model.hidden_layers is not None
    initial_hidden = model.hidden_layers[0].data.clone()

    train(model, dataloader, steps=config.steps, print_freq=1000)

    # Assert that the hidden layers are unchanged
    assert torch.allclose(model.hidden_layers[0].data, initial_hidden), "Hidden layer changed"


@pytest.mark.parametrize("n_hidden_layers", [0, 2])
def test_tms_spd_full_rank_equivalence(n_hidden_layers: int) -> None:
    """The full_rank SPD model with a single instance should have the same output and internal acts
    as the target model."""
    set_seed(0)

    batch_size = 4
    n_features = 3
    n_hidden = 2
    n_instances = 1
    k = 1  # Single subnetwork

    device = "cpu"

    # Create a target TMSModel
    target_model = TMSModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        device=device,
    )

    # Make the biases non-zero
    target_model.b_final.data = torch.randn_like(target_model.b_final.data)

    # Create the SPD model with k=1
    spd_model = TMSSPDFullRankModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        k=k,
        bias_val=0.0,
        device=device,
    )

    # Copy parameters from target model to SPD model
    spd_model.subnetwork_params.data[:, 0, :, :] = target_model.W.data
    spd_model.b_final.data[:, :] = target_model.b_final.data
    if spd_model.hidden_layers is not None:
        for i in range(n_hidden_layers):
            assert target_model.hidden_layers is not None
            spd_model.hidden_layers[i].subnetwork_params.data[:, 0, :, :] = (
                target_model.hidden_layers[i].data
            )

    # Create a random input
    input_data: Float[torch.Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, n_instances, n_features, device=device
    )

    # Forward pass through both models
    target_output, _, target_post_acts = target_model(input_data)
    spd_output, spd_layer_acts, _ = spd_model(input_data)

    # Assert outputs are the same
    assert torch.allclose(target_output, spd_output, atol=1e-6), "Outputs do not match"

    # Assert activations are the same for all layers
    for layer_name, target_act in target_post_acts.items():
        spd_act = spd_layer_acts[layer_name]
        assert torch.allclose(
            target_act, spd_act, atol=1e-6
        ), f"Activations do not match for layer {layer_name}"


def test_tms_spd_rank_penalty_full_rank_equivalence() -> None:
    """Test that TMSSPDRankPenaltyModel output and internal acts match TMSSPDFullRankModel.

    We set the bias as the same for both models.
    We set the A and B matrices of TMSSPDRankPenaltyModel to be the SVD of the
    subnetwork params of TMSSPDFullRankModel. We could instead use an identity and the original
    subnetwork params, but this is a slightly stronger test.

    """
    set_seed(0)

    batch_size = 4
    n_features = 3
    n_hidden = 2
    n_hidden_layers = 0  # This test will only work for 0 hidden layers
    n_instances = 2
    k = 2

    device = "cpu"

    # Create the full rank model
    full_rank_model = TMSSPDFullRankModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        k=k,
        bias_val=0.0,
        device=device,
    )

    # Create random parameters
    nn.init.xavier_normal_(full_rank_model.subnetwork_params)
    full_rank_model.b_final.data = torch.randn_like(full_rank_model.b_final.data)

    # Create the rank penalty model
    rank_penalty_model = TMSSPDRankPenaltyModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        n_hidden_layers=n_hidden_layers,
        k=k,
        bias_val=0.0,
        device=device,
    )

    # Copy bias
    rank_penalty_model.b_final.data = full_rank_model.b_final.data.clone()

    # For each instance and subnetwork, decompose the full rank parameters using SVD
    for i in range(n_instances):
        for j in range(k):
            W = full_rank_model.subnetwork_params[i, j]  # [n_features, n_hidden]
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            # Set A to U * sqrt(S) and B to sqrt(S) * Vh
            sqrt_S = torch.sqrt(S)
            # Note that since m = min(n_features, n_hidden) + 1, we need to add an extra column
            # of zeros to A and an extra row to B
            rank_penalty_model.A.data[i, j, :, -1] = 0
            rank_penalty_model.A.data[i, j, :, :-1] = U * sqrt_S.view(1, -1)  # [n_features, m]
            rank_penalty_model.B.data[i, j, -1, :] = 0
            rank_penalty_model.B.data[i, j, :-1, :] = sqrt_S.view(-1, 1) * Vh  # [m, n_hidden]

    # Create a random input
    input_data: Float[torch.Tensor, "batch n_instances n_features"] = torch.rand(
        batch_size, n_instances, n_features, device=device
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
