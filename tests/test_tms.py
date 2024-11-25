from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float

from spd.experiments.tms.models import (
    TMSModel,
    TMSSPDFullRankModel,
    TMSSPDModel,
    TMSSPDRankPenaltyModel,
)
from spd.experiments.tms.train_tms import TMSTrainConfig, train
from spd.experiments.tms.utils import TMSDataset
from spd.run_spd import Config, TMSConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple TMS config that we can use in multiple tests
TMS_TASK_CONFIG = TMSConfig(
    n_features=5,
    n_hidden=2,
    n_instances=2,
    k=5,
    feature_probability=0.5,
    train_bias=True,
    bias_val=0.0,
    pretrained_model_path=Path(""),  # We'll create this later
)


def tms_decomposition_optimize_test(config: Config):
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, TMSConfig)
    model = TMSSPDModel(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        n_hidden=config.task_config.n_hidden,
        k=config.task_config.k,
        bias_val=config.task_config.bias_val,
        device=device,
    )

    # For our pretrained model, just use a randomly initialized TMS model
    pretrained_model = TMSModel(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        n_hidden=config.task_config.n_hidden,
        device=device,
    )

    dataset = TMSDataset(
        n_instances=config.task_config.n_instances,
        n_features=config.task_config.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    # Pick an arbitrary parameter to check that it changes
    initial_param = model.A.clone().detach()

    if not config.task_config.train_bias:
        model.b_final.requires_grad = False

    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        param_map={"W": "W", "W_T": "W_T"},
        plot_results_fn=None,
    )

    assert not torch.allclose(
        initial_param, model.A
    ), "Model A matrix should have changed after optimization"


def test_tms_batch_topk_no_l2():
    config = Config(
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=None,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_tms_batch_topk_and_l2():
    config = Config(
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=0.1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_tms_topk_and_l2():
    config = Config(
        topk=2,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=0.1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_tms_lp():
    config = Config(
        topk=None,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        topk_l2_coeff=None,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_tms_topk_and_lp():
    config = Config(
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
        topk_l2_coeff=None,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_train_tms_happy_path():
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        n_features=3,
        n_hidden=2,
        n_instances=2,
        feature_probability=0.1,
        batch_size=32,
        steps=5,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
    )

    # Initialize model, dataset, and dataloader
    device = "cpu"
    model = TMSModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        n_hidden=config.n_hidden,
        device=device,
    )
    dataset = TMSDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out, _, _ = model(batch)
    initial_loss = torch.mean((labels.abs() - initial_out) ** 2)

    # Run optimize function
    train(model, dataloader, steps=config.steps, print_freq=1000)

    # Calculate final loss
    final_out, _, _ = model(batch)
    final_loss = torch.mean((labels.abs() - final_out) ** 2)

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Final loss ({final_loss:.2e}) is not lower than initial loss ({initial_loss:.2e})"


def test_tms_spd_full_rank_equivalence() -> None:
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
        device=device,
    )

    # Make the biases non-zero
    target_model.b_final.data = torch.randn_like(target_model.b_final.data)

    # Create the SPD model with k=1
    spd_model = TMSSPDFullRankModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        k=k,
        bias_val=0.0,
        device=device,
    )

    # Copy parameters from target model to SPD model
    spd_model.subnetwork_params.data[:, 0, :, :] = target_model.W.data
    spd_model.b_final.data[:, :] = target_model.b_final.data

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


def test_tms_dataset_one_feature_active():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5  # This won't be used when data_generation_type="exactly_one_active"
    device = "cpu"
    batch_size = 10

    dataset = TMSDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="exactly_one_active",
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that there's exactly one non-zero value per sample and instance
    for sample in batch:
        for instance in sample:
            non_zero_count = torch.count_nonzero(instance)
            assert non_zero_count == 1, f"Expected 1 non-zero value, but found {non_zero_count}"

    # Check that the non-zero values are between 0 and 1
    non_zero_values = batch[batch != 0]
    assert torch.all(
        (non_zero_values >= 0) & (non_zero_values <= 1)
    ), "Non-zero values should be between 0 and 1"


def test_tms_dataset_multi_feature():
    n_instances = 3
    n_features = 5
    feature_probability = 0.5
    device = "cpu"
    batch_size = 100

    dataset = TMSDataset(
        n_instances=n_instances,
        n_features=n_features,
        feature_probability=feature_probability,
        device=device,
        data_generation_type="at_least_zero_active",
    )

    batch, _ = dataset.generate_batch(batch_size)

    # Check shape
    assert batch.shape == (batch_size, n_instances, n_features), "Incorrect batch shape"

    # Check that the values are between 0 and 1
    assert torch.all((batch >= 0) & (batch <= 1)), "Values should be between 0 and 1"

    # Check that the proportion of non-zero elements is close to feature_probability
    non_zero_proportion = torch.count_nonzero(batch) / batch.numel()
    assert (
        abs(non_zero_proportion - feature_probability) < 0.05
    ), f"Expected proportion {feature_probability}, but got {non_zero_proportion}"


def test_set_full_rank_handcoded_spd_params():
    """Test that the full rank handcoded SPD model has the properties we'd expect.

    The properties we expect are:
    - The same output as the target model
    - The same internal acts as the target model
    - The subnetwork params are set such that only one row is non-zero for each subnetwork

    Note that the bias is currently not folded into the params.
    """
    set_seed(0)
    device = "cpu"
    n_instances = 2
    n_features = 5
    k = 5
    n_hidden = 3
    batch_size = 10

    # Create a target TMSModel
    target_model = TMSModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        device=device,
    )

    # Create the SPD model with k=n_features
    spd_model = TMSSPDFullRankModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
        k=k,
        bias_val=0.0,
        device=device,
    )

    # Set handcoded SPD params
    spd_model.set_handcoded_spd_params(target_model)

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

    # Check that the subnetwork params are set correctly
    for subnet_idx in range(k):
        # subnetwork_params is shape (n_instances, k, n_features, n_hidden) == (2, 5, 5, 3)
        # Check that only one row is non-zero for each subnetwork
        rows_not_zero = torch.any(spd_model.subnetwork_params.data[:, subnet_idx] != 0, dim=-1)
        assert torch.equal(
            torch.sum(rows_not_zero, dim=-1), torch.tensor([1, 1])
        ), f"Subnetwork {subnet_idx} should have only one non-zero row"

    # Check that the biases are the same (NOTE: currently not decomposing the biases)
    assert torch.allclose(
        spd_model.b_final.data, target_model.b_final.data, atol=1e-6
    ), "Biases do not match"


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
    n_instances = 2
    k = 2

    device = "cpu"

    # Create the full rank model
    full_rank_model = TMSSPDFullRankModel(
        n_instances=n_instances,
        n_features=n_features,
        n_hidden=n_hidden,
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
