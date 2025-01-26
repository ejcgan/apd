import torch
from jaxtyping import Float

from spd.experiments.piecewise.models import PiecewiseFunctionSPDTransformer
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.run_spd import (
    Config,
    PiecewiseConfig,
    optimize,
)
from spd.utils import BatchedDataLoader, calc_neuron_indices, set_seed


# Create a simple Piecewise config that we can use in multiple tests
def get_piecewise_config(n_layers: int = 2, simple_bias: bool = True) -> PiecewiseConfig:
    return PiecewiseConfig(
        n_functions=5,  # Only works when n_functions==k-1. TODO: Verify that we want this.
        neurons_per_function=10,
        n_layers=n_layers,
        feature_probability=0.1,
        range_min=0.0,
        range_max=5.0,
        k=6,
        simple_bias=simple_bias,
    )


def piecewise_decomposition_optimize_test(config: Config, check_A_changed: bool = True) -> None:
    set_seed(0)
    device = "cpu"
    assert isinstance(config.task_config, PiecewiseConfig)

    piecewise_model, piecewise_model_spd, dataloader = get_model_and_dataloader(config, device)[:3]

    # Pick an arbitrary parameter to check that it changes
    initial_param: Float[torch.Tensor, " d_mlp"] = (
        piecewise_model_spd.mlps[0].linear1.A.clone().detach()
    )
    # Params that should stay the same:
    inital_bias1_vals = [
        piecewise_model_spd.mlps[i].linear1.bias for i in range(len(piecewise_model_spd.mlps))
    ]
    initial_W_E = piecewise_model_spd.W_E.weight.clone().detach()
    initial_W_U = piecewise_model_spd.W_U.weight.clone().detach()

    param_map = {}
    for i in range(piecewise_model_spd.n_layers):
        param_map[f"mlp_{i}.input_layer.weight"] = f"mlp_{i}.input_layer.weight"
        param_map[f"mlp_{i}.output_layer.weight"] = f"mlp_{i}.output_layer.weight"

    assert isinstance(piecewise_model_spd, PiecewiseFunctionSPDTransformer)
    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=piecewise_model,
        param_map=param_map,
        plot_results_fn=None,
    )

    if check_A_changed:
        assert not torch.allclose(
            initial_param, piecewise_model_spd.mlps[0].linear1.A
        ), "Model A matrix should have changed after optimization"

    # Check that other params are exactly equal
    for i in range(len(piecewise_model_spd.mlps)):
        assert torch.allclose(piecewise_model_spd.mlps[i].linear1.bias, inital_bias1_vals[i])
    assert torch.allclose(piecewise_model_spd.W_E.weight, initial_W_E)
    assert torch.allclose(piecewise_model_spd.W_U.weight, initial_W_U)


def test_piecewise_batch_topk_no_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=4,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_batch_topk_and_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=4,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_topk_and_l2() -> None:
    config = Config(
        topk=4,
        batch_topk=False,
        batch_size=4,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_lp() -> None:
    config = Config(
        topk=None,
        batch_topk=False,
        batch_size=4,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        task_config=get_piecewise_config(),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_lp_simple_bias_false() -> None:
    config = Config(
        topk=None,
        batch_topk=False,
        batch_size=4,
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        lp_sparsity_coeff=0.01,
        pnorm=0.9,
        task_config=get_piecewise_config(simple_bias=False),
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
        n_inputs=n_inputs,
        functions=functions,
        feature_probability=feature_probability,
        range_min=range_min,
        range_max=range_max,
        batch_size=batch_size,
        return_labels=True,
    )

    # Create dataloader
    dataloader = BatchedDataLoader(dataset)

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


def test_calc_neuron_indices():
    neuron_permutations = (torch.tensor([8, 6, 2, 11, 0, 5]), torch.tensor([1, 9, 3, 10, 7, 4]))
    neurons_per_function = 3
    num_functions = 4
    indices = calc_neuron_indices(neuron_permutations, neurons_per_function, num_functions)
    expected_indices = [
        [
            torch.tensor([2, 4]),
            torch.tensor([5]),
            torch.tensor([0, 1]),
            torch.tensor([3]),
        ],
        [
            torch.tensor([0]),
            torch.tensor([2, 5]),
            torch.tensor([4]),
            torch.tensor([1, 3]),
        ],
    ]
    for i in range(2):
        for j in range(4):
            torch.testing.assert_close(indices[i][j], expected_indices[i][j])
