import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.run_spd import Config, PiecewiseConfig, optimize
from spd.utils import set_seed


# Create a simple Piecewise config that we can use in multiple tests
def get_piecewise_config(handcoded_AB: bool = False, n_layers: int = 2) -> PiecewiseConfig:
    return PiecewiseConfig(
        n_functions=5,  # Only works when n_functions==k-1. TODO: Verify that we want this.
        neurons_per_function=10,
        n_layers=n_layers,
        feature_probability=0.1,
        range_min=0.0,
        range_max=5.0,
        k=6,
        handcoded_AB=handcoded_AB,
    )


def piecewise_decomposition_optimize_test(config: Config) -> None:
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, PiecewiseConfig)

    piecewise_model, piecewise_model_spd, dataloader = get_model_and_dataloader(config, device)

    # Pick an arbitrary parameter to check that it changes
    initial_param: Float[torch.Tensor, " d_mlp"] = (
        piecewise_model_spd.mlps[0].linear1.A.clone().detach()
    )
    # Params that should stay the same:
    inital_bias1_vals = [
        piecewise_model_spd.mlps[i].bias1 for i in range(len(piecewise_model_spd.mlps))
    ]
    initial_W_E = piecewise_model_spd.W_E.weight.clone().detach()
    initial_W_U = piecewise_model_spd.W_U.weight.clone().detach()

    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=piecewise_model,
        plot_results_fn=None,
    )

    assert not torch.allclose(
        initial_param, piecewise_model_spd.mlps[0].linear1.A
    ), "Model A matrix should have changed after optimization"

    # Check that other params are exactly equal
    for i in range(len(piecewise_model_spd.mlps)):
        assert torch.allclose(piecewise_model_spd.mlps[i].bias1, inital_bias1_vals[i])
    assert torch.allclose(piecewise_model_spd.W_E.weight, initial_W_E)
    assert torch.allclose(piecewise_model_spd.W_U.weight, initial_W_U)


def test_piecewise_batch_tokp_no_l2_handcoded_AB() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=2,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=None,
        task_config=get_piecewise_config(handcoded_AB=True, n_layers=1),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_batch_topk_no_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=2,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=None,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_batch_topk_and_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=2,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=0.1,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_topk_and_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=False,
        batch_size=2,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=0.1,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_lp() -> None:
    config = Config(
        topk=None,
        batch_topk=False,
        batch_size=2,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        topk_l2_coeff=None,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_dataset():
    set_seed(0)
    # Define test parameters
    n_inputs = 5
    functions = [lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: torch.sin(x)]
    feature_probability = 0.5
    range_min = 0
    range_max = 5
    batch_size = 10

    # Create dataset
    dataset = PiecewiseDataset(
        n_inputs, functions, feature_probability, range_min, range_max, buffer_size=8
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Get a batch of samples
    batch_x, batch_y = next(iter(dataloader))

    # Check shape
    assert batch_x.shape == (batch_size, n_inputs)
    assert batch_y.shape == (batch_size, 1)

    # Check first column (real values)
    assert torch.all((batch_x[:, 0] >= 0) & (batch_x[:, 0] <= range_max))

    # Check control bits (all but the last sample)
    control_bits = batch_x[:-1, 1:]
    assert torch.all((control_bits == 0) | (control_bits == 1))

    # Check that there is a non-zero input for each batch element
    assert torch.all(control_bits.any(dim=1))

    # The mean of the control bits should be >= feature_probability (greater because we remove
    # all rows with all zeros). We allow a small difference as we're only using batch_size=10.
    mean_control_bits = control_bits.float().mean()
    assert mean_control_bits >= (feature_probability - 0.1)
