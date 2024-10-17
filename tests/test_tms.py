import torch
from jaxtyping import Float

from spd.experiments.tms.models import TMSModel, TMSSPDFullRankModel, TMSSPDModel
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
    pretrained_model_path=None,  # We'll create this later
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
        train_bias=config.task_config.train_bias,
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
        n_features=3, n_hidden=2, n_instances=2, feature_probability=0.1, batch_size=32, steps=5
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
        train_bias=True,
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
