import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from spd.experiments.piecewise.models import (
    PiecewiseFunctionSPDFullRankTransformer,
    PiecewiseFunctionSPDRankPenaltyTransformer,
    PiecewiseFunctionSPDTransformer,
    PiecewiseFunctionTransformer,
)
from spd.experiments.piecewise.piecewise_dataset import PiecewiseDataset
from spd.experiments.piecewise.piecewise_decomposition import get_model_and_dataloader
from spd.run_spd import (
    Config,
    PiecewiseConfig,
    calc_param_match_loss,
    calc_recon_mse,
    optimize,
)
from spd.utils import (
    BatchedDataLoader,
    calc_grad_attributions_rank_one,
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
        piecewise_model_spd.mlps[i].bias1 for i in range(len(piecewise_model_spd.mlps))
    ]
    initial_W_E = piecewise_model_spd.W_E.weight.clone().detach()
    initial_W_U = piecewise_model_spd.W_U.weight.clone().detach()

    param_map = {}
    for i in range(piecewise_model_spd.n_layers):
        param_map[f"mlp_{i}.input_layer.weight"] = f"mlp_{i}.input_layer.weight"
        param_map[f"mlp_{i}.output_layer.weight"] = f"mlp_{i}.output_layer.weight"

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
        assert torch.allclose(piecewise_model_spd.mlps[i].bias1, inital_bias1_vals[i])
    assert torch.allclose(piecewise_model_spd.W_E.weight, initial_W_E)
    assert torch.allclose(piecewise_model_spd.W_U.weight, initial_W_U)


def test_piecewise_batch_topk_no_l2_handcoded_AB() -> None:
    config = Config(
        topk=4,
        batch_topk=True,
        batch_size=4,  # Needs to be enough to have at least 1 control bit on in two steps
        steps=2,
        print_freq=2,
        save_freq=None,
        lr=1e-3,
        topk_recon_coeff=1,
        topk_l2_coeff=None,
        task_config=get_piecewise_config(handcoded_AB=True, n_layers=1),
    )
    piecewise_decomposition_optimize_test(config, check_A_changed=False)


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
        topk_l2_coeff=None,
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
        topk_l2_coeff=0.1,
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
        topk_l2_coeff=0.1,
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
        topk_l2_coeff=None,
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
        topk_l2_coeff=None,
        task_config=get_piecewise_config(simple_bias=False),
    )
    piecewise_decomposition_optimize_test(config)


def test_piecewise_batch_topk_rank_one_simple_bias_false_loss_stable() -> None:
    """After training for a few steps, topk_recon_loss and param_match_loss should stay at ~0."""
    set_seed(0)
    device = "cpu"
    config = Config(
        seed=0,
        spd_type="rank_one",
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
    # Rank 1 case
    assert isinstance(piecewise_model_spd, PiecewiseFunctionSPDTransformer)
    piecewise_model.requires_grad_(False)
    piecewise_model.to(device=device)

    batch = next(iter(dataloader))[0]
    batch = batch.to(device=device)
    with torch.inference_mode():
        labels, _, _ = piecewise_model(batch)

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
        out_topk, _, inner_acts_topk = piecewise_model_spd(batch, topk_mask=topk_mask)
        assert len(inner_acts_topk) == piecewise_model_spd.n_param_matrices

        initial_topk_recon_loss = calc_recon_mse(out_topk, labels, has_instance_dim=False)
        return initial_topk_recon_loss

    # Get initial losses (param_match_loss and topk_recon_loss)
    # Initial param match loss
    pretrained_weights = piecewise_model.all_decomposable_params()

    n_params = sum(p.numel() for p in pretrained_weights.values())
    param_map = {}
    for i in range(piecewise_model_spd.n_layers):
        param_map[f"mlp_{i}.input_layer.weight"] = f"mlp_{i}.input_layer.weight"
        param_map[f"mlp_{i}.output_layer.weight"] = f"mlp_{i}.output_layer.weight"

    initial_param_match_loss = calc_param_match_loss(
        pretrained_weights=pretrained_weights,
        subnetwork_params_summed=piecewise_model_spd.all_subnetwork_params_summed(),
        param_map=param_map,
        has_instance_dim=False,
        n_params=n_params,
    )

    # Rank 1 so layer_acts is None
    attribution_scores = calc_grad_attributions_rank_one(
        out=out, inner_acts_vals=list(inner_acts.values())
    )
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
        param_map=param_map,
        plot_results_fn=None,
    )

    # Check that the losses have not reduced
    final_param_match_loss = calc_param_match_loss(
        pretrained_weights=pretrained_weights,
        subnetwork_params_summed=piecewise_model_spd.all_subnetwork_params_summed(),
        param_map=param_map,
        has_instance_dim=False,
        n_params=n_params,
    )

    out, _, inner_acts = piecewise_model_spd(batch)
    attribution_scores = calc_grad_attributions_rank_one(
        out=out, inner_acts_vals=list(inner_acts.values())
    )
    final_topk_recon_loss = get_topk_recon_on_batch(
        batch, labels, attribution_scores, piecewise_model_spd
    )

    # Check that the final losses are still small
    # TODO: When we have GPU tests, run more steps and lower the precision. Current test only shows
    # that the loss doesn't blow up dramatically straight away.
    assert final_param_match_loss < 1e-3
    assert final_topk_recon_loss < 1e-2


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


def test_piecewise_spd_full_rank_equivalence() -> None:
    device = "cpu"
    set_seed(0)

    batch_size = 4
    n_inputs = 4  # 3 functions + 1 input
    d_mlp = 6
    n_layers = 2
    k = 1  # Single subnetwork

    # Create a target PiecewiseFunctionTransformer
    target_model = PiecewiseFunctionTransformer(
        n_inputs=n_inputs, d_mlp=d_mlp, n_layers=n_layers
    ).to(device)

    # Init all params to random values
    for name, param in target_model.named_parameters():
        # Except for the output_layer biases which should remain zero (that's the standard
        # setup we're using for piecewise)
        if "output_layer.bias" in name:
            param.data = torch.zeros_like(param.data)
        else:
            param.data = torch.randn_like(param.data)

    # Create the SPD model with k=1
    spd_model = PiecewiseFunctionSPDFullRankTransformer(
        n_inputs=n_inputs, d_mlp=d_mlp, n_layers=n_layers, k=k, init_scale=1.0
    ).to(device)

    # Copy parameters from target model to SPD model
    spd_model.W_E.weight.data = target_model.W_E.weight.data
    spd_model.W_U.weight.data = target_model.W_U.weight.data
    spd_model.set_subnet_to_target(target_model, dim=0)

    # Create a random input
    input_data: Float[torch.Tensor, "batch n_inputs"] = torch.rand(
        batch_size, n_inputs, device=device
    )

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


def test_piecewise_spd_rank_penalty_rank_one_equivalence() -> None:
    """Test that PiecewiseFunctionSPDTransformer output and internal acts match
    PiecewiseFunctionSPDRankPenaltyTransformer when m=1.

    We set the bias and embeddings as the same for both models.
    We directly copy the A and B matrices from the SPD model to the rank penalty model
    since m=1 makes them equivalent.
    """
    set_seed(0)

    batch_size = 4
    n_inputs = 3
    d_mlp = 8
    n_layers = 1
    k = 4
    m = 1
    init_scale = 1.0

    device = "cpu"

    # Create the SPD model
    spd_model = PiecewiseFunctionSPDTransformer(
        n_inputs=n_inputs,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=k,
        init_scale=init_scale,
    ).to(device)

    # Randomly initialize params again to avoid zeros
    for param in spd_model.parameters():
        if param.dim() >= 2:
            nn.init.xavier_normal_(param)
        else:
            # # For 1D parameters (biases), initialize with small random values
            nn.init.uniform_(param, -0.1, 0.1)

    # Create the rank penalty model with m=1
    rank_penalty_model = PiecewiseFunctionSPDRankPenaltyTransformer(
        n_inputs=n_inputs,
        d_mlp=d_mlp,
        n_layers=n_layers,
        k=k,
        init_scale=init_scale,
        m=m,
    ).to(device)

    # Copy embedding matrices
    rank_penalty_model.W_E.weight.data[:] = spd_model.W_E.weight.data.clone()
    rank_penalty_model.W_U.weight.data[:] = spd_model.W_U.weight.data.clone()

    # For each MLP layer, copy the A and B matrices and biases
    for i in range(n_layers):
        # Copy biases
        rank_penalty_model.mlps[i].linear1.bias.data[:] = spd_model.mlps[i].bias1.data.clone()

        # Copy A and B matrices for input layer
        rank_penalty_model.mlps[i].linear1.A.data[:, :, :] = einops.rearrange(
            spd_model.mlps[i].linear1.A.data, "d_embed k -> k d_embed 1"
        )
        rank_penalty_model.mlps[i].linear1.B.data[:, :, :] = einops.rearrange(
            spd_model.mlps[i].linear1.B.data, "k d_mlp -> k 1 d_mlp"
        )

        # Copy A and B matrices for output layer
        rank_penalty_model.mlps[i].linear2.A.data[:, :, :] = einops.rearrange(
            spd_model.mlps[i].linear2.A.data, "d_mlp k -> k d_mlp 1"
        )
        rank_penalty_model.mlps[i].linear2.B.data[:, :, :] = einops.rearrange(
            spd_model.mlps[i].linear2.B.data, "k d_embed -> k 1 d_embed"
        )

    # Create a random input
    input_data: Float[torch.Tensor, "batch n_inputs"] = torch.rand(
        batch_size, n_inputs, device=device
    )

    # Forward pass through both models
    spd_output, spd_layer_acts, spd_inner_acts = spd_model(input_data)
    rank_penalty_output, rank_penalty_layer_acts, rank_penalty_inner_acts = rank_penalty_model(
        input_data
    )

    # Assert outputs are the same
    assert torch.allclose(spd_output, rank_penalty_output, atol=1e-6), "Outputs do not match"

    # Assert activations are the same for all layers
    for layer_name in spd_layer_acts:
        spd_act = spd_layer_acts[layer_name]
        rank_penalty_act = rank_penalty_layer_acts[layer_name]
        assert torch.allclose(
            spd_act, rank_penalty_act, atol=1e-6
        ), f"Layer activations do not match for layer {layer_name}"
