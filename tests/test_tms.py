import torch

from spd.run_spd import Config, TMSConfig, optimize
from spd.scripts.tms.models import TMSModel, TMSSPDModel
from spd.scripts.tms.train_tms import TMSTrainConfig, train
from spd.scripts.tms.utils import TMSDataset
from spd.utils import BatchedDataLoader, set_seed

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
    dataloader = BatchedDataLoader(dataset, batch_size=config.batch_size)

    # Pick an arbitrary parameter to check that it changes
    initial_param = model.A.clone().detach()

    optimize(
        model=model,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=pretrained_model,
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
        max_sparsity_coeff=1,
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
        max_sparsity_coeff=1,
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
        max_sparsity_coeff=1,
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
        max_sparsity_coeff=0.01,
        pnorm=0.9,
        topk_l2_coeff=None,
        task_config=TMS_TASK_CONFIG,
    )
    tms_decomposition_optimize_test(config)


def test_train_tms_happy_path():
    set_seed(0)
    # Set up a small configuration
    config = TMSTrainConfig(
        n_features=3, n_hidden=2, n_instances=2, feature_probability=0.1, batch_size=32
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
    dataloader = BatchedDataLoader(dataset, batch_size=config.batch_size)

    # Calculate initial loss
    batch, labels = next(iter(dataloader))
    initial_out = model(batch)
    initial_loss = torch.mean((labels.abs() - initial_out) ** 2)

    # Run optimize function
    train(model, dataloader, steps=5, print_freq=1000)

    # Calculate final loss
    final_out = model(batch)
    final_loss = torch.mean((labels.abs() - final_out) ** 2)

    # Assert that the final loss is lower than the initial loss
    assert (
        final_loss < initial_loss
    ), f"Final loss ({final_loss:.4f}) is not lower than initial loss ({initial_loss:.4f})"
