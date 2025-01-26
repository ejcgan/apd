from pathlib import Path

import pytest
import torch

from spd.experiments.tms.models import (
    TMSModel,
    TMSModelConfig,
    TMSSPDModel,
    TMSSPDModelConfig,
)
from spd.experiments.tms.train_tms import TMSTrainConfig, get_model_and_dataloader, train
from spd.run_spd import Config, TMSTaskConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, SparseFeatureDataset, set_seed

# Create a simple TMS config that we can use in multiple tests
TMS_TASK_CONFIG = TMSTaskConfig(
    task_name="tms",
    k=5,
    feature_probability=0.5,
    train_bias=False,
    bias_val=0.0,
    pretrained_model_path=Path(""),  # We'll create this later
)


def tms_spd_happy_path(config: Config, n_hidden_layers: int = 0):
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, TMSTaskConfig)

    # For our pretrained model, just use a randomly initialized TMS model
    tms_model_config = TMSModelConfig(
        n_instances=2,
        n_features=5,
        n_hidden=2,
        n_hidden_layers=n_hidden_layers,
        device=device,
    )
    target_model = TMSModel(config=tms_model_config)

    tms_spd_model_config = TMSSPDModelConfig(
        **tms_model_config.model_dump(mode="json"),
        k=config.task_config.k,
        bias_val=config.task_config.bias_val,
    )
    model = TMSSPDModel(config=tms_spd_model_config)
    # Randomly initialize the bias for the pretrained model
    target_model.b_final.data = torch.randn_like(target_model.b_final.data)
    # Manually set the bias for the SPD model from the bias in the pretrained model
    model.b_final.data[:] = target_model.b_final.data.clone()

    if not config.task_config.train_bias:
        model.b_final.requires_grad = False

    dataset = SparseFeatureDataset(
        n_instances=target_model.config.n_instances,
        n_features=target_model.config.n_features,
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
        pretrained_model=target_model,
        param_map=param_map,
        plot_results_fn=None,
    )

    assert not torch.allclose(
        initial_param, model.A
    ), "Model A matrix should have changed after optimization"


def test_tms_batch_topk_no_schatten():
    config = Config(
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
    tms_spd_happy_path(config)


@pytest.mark.parametrize("n_hidden_layers", [0, 2])
def test_tms_batch_topk_and_schatten(n_hidden_layers: int):
    config = Config(
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
    tms_spd_happy_path(config, n_hidden_layers)


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
        schatten_pnorm=0.9,
        schatten_coeff=1e-1,
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_happy_path(config)


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
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_happy_path(config)


@pytest.mark.parametrize("n_hidden_layers", [0, 2])
def test_tms_topk_and_lp(n_hidden_layers: int):
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
        task_config=TMS_TASK_CONFIG,
    )
    tms_spd_happy_path(config, n_hidden_layers)


def test_train_tms_happy_path():
    device = "cpu"
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=0,
            device=device,
        ),
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

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

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
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=2,
            device=device,
        ),
        feature_probability=0.1,
        batch_size=32,
        steps=2,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
        fixed_identity_hidden_layers=True,
        fixed_random_hidden_layers=False,
    )

    model, dataloader = get_model_and_dataloader(config, device)

    eye = torch.eye(config.tms_model_config.n_hidden, device=device).expand(
        config.tms_model_config.n_instances, -1, -1
    )

    assert model.hidden_layers is not None
    # Assert that this is an identity matrix
    initial_hidden = model.hidden_layers[0].data.clone()
    assert torch.allclose(initial_hidden, eye), "Initial hidden layer is not identity"

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

    # Assert that the hidden layers remains identity
    assert torch.allclose(model.hidden_layers[0].data, eye), "Hidden layer changed"


def test_tms_train_fixed_random():
    """Check that hidden layer is random before and after training."""
    device = "cpu"
    set_seed(0)
    config = TMSTrainConfig(
        tms_model_config=TMSModelConfig(
            n_features=3,
            n_hidden=2,
            n_instances=2,
            n_hidden_layers=2,
            device=device,
        ),
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

    train(model, dataloader, steps=config.steps, print_freq=1000, log_wandb=False)

    # Assert that the hidden layers are unchanged
    assert torch.allclose(model.hidden_layers[0].data, initial_hidden), "Hidden layer changed"
