import torch

from spd.experiments.linear.linear_dataset import DeepLinearDataset
from spd.experiments.linear.models import DeepLinearComponentModel, DeepLinearModel
from spd.experiments.linear.train_linear import Config as TrainConfig
from spd.experiments.linear.train_linear import train
from spd.run_spd import Config, DeepLinearConfig, optimize
from spd.utils import DatasetGeneratedDataLoader, set_seed

# Create a simple DeepLinear config that we can use in multiple tests
DEEP_LINEAR_TASK_CONFIG = DeepLinearConfig(
    n_features=5,
    n_layers=2,
    n_instances=2,
    k=5,
    pretrained_model_path=None,  # We'll create this later
)


def deep_linear_decomposition_optimize_test(config: Config) -> None:
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, DeepLinearConfig)

    assert config.task_config.n_features is not None
    assert config.task_config.n_layers is not None
    assert config.task_config.n_instances is not None

    # For our pretrained model, just use a randomly initialized DeepLinear model
    pretrained_model = DeepLinearModel(
        n_features=config.task_config.n_features,
        n_layers=config.task_config.n_layers,
        n_instances=config.task_config.n_instances,
    ).to(device)

    model = DeepLinearComponentModel(
        n_features=config.task_config.n_features,
        n_layers=config.task_config.n_layers,
        n_instances=config.task_config.n_instances,
        k=config.task_config.k,
    ).to(device)

    dataset = DeepLinearDataset(config.task_config.n_features, config.task_config.n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)

    # Pick an arbitrary parameter to check that it changes
    initial_param = model.layers[0].A.clone().detach()

    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
        param_map={f"layer_{i}": f"layer_{i}" for i in range(config.task_config.n_layers)},
        plot_results_fn=None,
    )

    assert not torch.allclose(
        initial_param, model.layers[0].A
    ), "Model A matrix should have changed after optimization"


def test_deep_linear_batch_topk_no_l2() -> None:
    config = Config(
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=0.01,
        topk_l2_coeff=None,
        task_config=DEEP_LINEAR_TASK_CONFIG,
    )
    deep_linear_decomposition_optimize_test(config)


def test_deep_linear_batch_topk_and_l2() -> None:
    config = Config(
        topk=2,
        batch_topk=True,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=0.01,
        topk_l2_coeff=0.1,
        task_config=DEEP_LINEAR_TASK_CONFIG,
    )
    deep_linear_decomposition_optimize_test(config)


def test_deep_linear_batch_topk_and_lp_and_l2() -> None:
    config = Config(
        topk=2,
        batch_topk=True,
        pnorm=0.9,
        lp_sparsity_coeff=0.01,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=0.01,
        topk_l2_coeff=0.1,
        task_config=DEEP_LINEAR_TASK_CONFIG,
    )
    deep_linear_decomposition_optimize_test(config)


def test_deep_linear_topk_and_l2() -> None:
    config = Config(
        topk=2,
        batch_topk=False,
        batch_size=4,
        steps=4,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=0.01,
        topk_l2_coeff=0.1,
        task_config=DEEP_LINEAR_TASK_CONFIG,
    )
    deep_linear_decomposition_optimize_test(config)


def test_deep_linear_lp() -> None:
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
        task_config=DEEP_LINEAR_TASK_CONFIG,
    )
    deep_linear_decomposition_optimize_test(config)


def test_train_linear_happy_path() -> None:
    set_seed(0)
    device = "cpu"
    config = TrainConfig(
        n_features=2,
        n_layers=2,
        n_instances=2,
        batch_size=2,
        steps=3,  # Run only a few steps
        print_freq=100,
        lr=0.01,
    )

    model = DeepLinearModel(config.n_features, config.n_layers, config.n_instances).to(device)
    dataset = DeepLinearDataset(config.n_features, config.n_instances)
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out = model(batch.to(device))
    initial_loss = torch.mean((labels.to(device) - initial_out) ** 2).item()

    # Train the model
    train(config, model, dataloader, device, out_dir=None)

    # Calculate final loss
    final_out = model(batch.to(device))
    final_loss = torch.mean((labels.to(device) - final_out) ** 2).item()

    assert (
        final_loss < initial_loss
    ), f"Expected final loss to be lower than initial loss, but got {final_loss} >= {initial_loss}"
