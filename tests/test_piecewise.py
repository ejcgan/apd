import torch
from jaxtyping import Float

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.run_spd import Config, PiecewiseConfig, calc_param_match_loss, calc_recon_mse, optimize
from spd.utils import (
    BatchedDataLoader,
    calc_attributions,
    calc_neuron_indices,
    calc_topk_mask,
    set_seed,
)


# Create a simple Piecewise config that we can use in multiple tests
def get_piecewise_config(
    handcoded_AB: bool = False, n_layers: int = 2, simple_bias: bool = True
) -> PiecewiseConfig:
    return PiecewiseConfig(
        n_functions=5,  # Only works when n_functions==k-1. TODO: Verify that we want this.
        neurons_per_function=10,
        n_layers=n_layers,
        feature_probability=0.1,
        range_min=0.0,
        range_max=5.0,
        k=6,
        handcoded_AB=handcoded_AB,
        simple_bias=simple_bias,
    )


def piecewise_decomposition_optimize_test(config: Config) -> None:
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


def test_piecewise_lp_simple_bias_false() -> None:
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
        task_config=get_piecewise_config(simple_bias=False),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_batch_topk_simple_bias_false_loss_stable() -> None:
    """After training for a few steps, the topk_recon_loss and param_match_loss should stay at 0."""
    set_seed(0)
    device = "cpu"
    config = Config(
        topk=2,
        batch_topk=True,
        batch_size=256,
        steps=100,
        print_freq=20,
        save_freq=None,
        lr=1e-2,
        topk_recon_coeff=0.1,
        lp_sparsity_coeff=None,
        pnorm=None,
        topk_l2_coeff=None,
        task_config=get_piecewise_config(
            handcoded_AB=True,
            n_layers=1,
            simple_bias=False,
        ),
    )
    assert isinstance(config.task_config, PiecewiseConfig)

    piecewise_model, piecewise_model_spd, dataloader = get_model_and_dataloader(config, device)[:3]
    piecewise_model.requires_grad_(False)
    piecewise_model.to(device=device)

    batch = next(iter(dataloader))[0]
    batch = batch.to(device=device)
    with torch.inference_mode():
        labels = piecewise_model(batch)

    out, _, inner_acts = piecewise_model_spd(batch)

    def get_topk_recon_on_batch(
        batch: Float[torch.Tensor, "batch_size input_dim"],
        labels: Float[torch.Tensor, " batch_size 1"],
        attribution_scores: Float[torch.Tensor, "batch_size ... k"],
        piecewise_model_spd: PiecewiseFunctionSPDTransformer,
    ) -> Float[torch.Tensor, ""]:
        assert config.topk is not None
        topk_mask = calc_topk_mask(attribution_scores, config.topk, batch_topk=config.batch_topk)

        # Do a forward pass with only the topk subnetworks
        out_topk, _, inner_acts_topk = piecewise_model_spd.forward_topk(batch, topk_mask=topk_mask)
        assert len(inner_acts_topk) == piecewise_model_spd.n_param_matrices

        initial_topk_recon_loss = calc_recon_mse(out_topk, labels, has_instance_dim=False)
        return initial_topk_recon_loss

    # Get initial losses (param_match_loss and topk_recon_loss)
    # Initial param match loss
    pretrained_weights = piecewise_model.all_decomposable_params()
    initial_param_match_loss = calc_param_match_loss(
        model=piecewise_model_spd, pretrained_weights=pretrained_weights, device=device
    )

    attribution_scores = calc_attributions(out, inner_acts)
    initial_topk_recon_loss = get_topk_recon_on_batch(
        batch, labels, attribution_scores, piecewise_model_spd
    )

    # Check that initial losses are small
    assert initial_param_match_loss < 1e-3
    assert initial_topk_recon_loss < 1e-3

    # Get initial topk recon loss
    optimize(
        model=piecewise_model_spd,
        config=config,
        out_dir=None,
        device=device,
        dataloader=dataloader,
        pretrained_model=piecewise_model,
        plot_results_fn=None,
    )

    # Check that the losses have not reduced
    final_param_match_loss = calc_param_match_loss(
        model=piecewise_model_spd, pretrained_weights=pretrained_weights, device=device
    )

    out, _, inner_acts = piecewise_model_spd(batch)
    attribution_scores = calc_attributions(out, inner_acts)
    final_topk_recon_loss = get_topk_recon_on_batch(
        batch, labels, attribution_scores, piecewise_model_spd
    )

    # Check that the final losses are still small
    # TODO: When we have GPU tests, run more steps and lower the precision. Current test only shows
    # that the loss doesn't blow up dramatically straight away.
    assert final_param_match_loss < 1e-3
    assert final_topk_recon_loss < 3e-3


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
