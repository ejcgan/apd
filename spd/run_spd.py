"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.log import logger
from spd.models.base import Model, SPDFullRankModel, SPDModel
from spd.types import Probability, RootPath
from spd.utils import (
    calc_ablation_attributions,
    calc_activation_attributions,
    calc_grad_attributions_full_rank,
    calc_grad_attributions_rank_one,
    calc_topk_mask,
)


class TMSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = "tms"
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_instances: PositiveInt
    k: PositiveInt
    feature_probability: Probability
    train_bias: bool
    bias_val: float
    pretrained_model_path: RootPath | None = None


class DeepLinearConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["deep_linear"] = "deep_linear"
    n_features: PositiveInt | None = None
    n_layers: PositiveInt | None = None
    n_instances: PositiveInt | None = None
    k: PositiveInt | None = None
    pretrained_model_path: RootPath | None = None


class PiecewiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["piecewise"] = "piecewise"
    n_functions: PositiveInt
    neurons_per_function: PositiveInt
    n_layers: PositiveInt
    feature_probability: Probability
    range_min: float
    range_max: float
    k: PositiveInt
    init_scale: float = 1.0
    target_seed: int | None = None
    dataset_seed: int | None = None
    simple_bias: bool = False
    handcoded_AB: bool = False
    decompose_bias: bool = True


class ResidualLinearConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_linear"] = "residual_linear"
    k: PositiveInt
    feature_probability: Probability
    init_scale: float = 1.0
    one_feature_active: bool = False
    pretrained_model_path: RootPath


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    full_rank: bool = False
    seed: int = 0
    topk: PositiveFloat | None = None
    batch_topk: bool = True
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    image_freq: PositiveInt | None = None
    slow_images: bool = False
    save_freq: PositiveInt | None = None
    lr: PositiveFloat
    orthog_coeff: NonNegativeFloat | None = None
    out_recon_coeff: NonNegativeFloat | None = None
    topk_act_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    topk_recon_coeff: NonNegativeFloat | None = None
    topk_l2_coeff: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat | None = None
    topk_param_attrib_coeff: NonNegativeFloat | None = None
    distil_from_target: bool = False
    pnorm: PositiveFloat | None = None
    pnorm_end: PositiveFloat | None = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    sparsity_warmup_pct: Probability = 0.0
    unit_norm_matrices: bool = True
    attribution_type: Literal["gradient", "ablation", "activation"] = "gradient"
    task_config: DeepLinearConfig | PiecewiseConfig | TMSConfig | ResidualLinearConfig = Field(
        ..., discriminator="task_name"
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check valid combinations of topk and batch_size
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer():
                    logger.warning(
                        f"batch_size * topk={self.batch_size * self.topk} is not an integer, will "
                        f"round down from {self.batch_size * self.topk} to "
                        f"{int(self.batch_size * self.topk)} when calculating topk_mask"
                    )
            else:
                if not self.topk.is_integer():
                    raise ValueError("topk must be an integer when not using batch_topk")

        # Warn if neither topk_recon_coeff nor lp_sparsity_coeff is set
        if not self.topk_recon_coeff and not self.lp_sparsity_coeff:
            logger.warning("Neither topk_recon_coeff nor lp_sparsity_coeff is set")

        # If topk_recon_coeff is set, topk must be set
        if self.topk_recon_coeff is not None:
            assert self.topk is not None, "topk must be set if topk_recon_coeff is set"

        # If lp_sparsity_coeff is set, pnorm or pnorm_end must be set
        if self.lp_sparsity_coeff is not None:
            assert (
                self.pnorm is not None or self.pnorm_end is not None
            ), "pnorm or pnorm_end must be set if lp_sparsity_coeff is set"

        # Check that topk_l2_coeff and topk_recon_coeff are None if topk is None
        if self.topk is None:
            assert self.topk_l2_coeff is None, "topk_l2_coeff is not None but topk is"
            assert self.topk_recon_coeff is None, "topk_recon_coeff is not None but topk is"

        # Give a warning if both out_recon_coeff and param_match_coeff are > 0
        if (
            self.param_match_coeff is not None
            and self.param_match_coeff > 0
            and self.out_recon_coeff is not None
            and self.out_recon_coeff > 0
        ):
            logger.warning(
                "Both param_match_coeff and out_recon_coeff are > 0. It's typical to only set one."
            )

        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.topk_l2_coeff == 0:
            logger.warning(f"topk_l2_coeff {msg}")
        if self.topk_recon_coeff == 0:
            logger.warning(f"topk_recon_coeff {msg}")
        if self.lp_sparsity_coeff == 0:
            logger.warning(f"lp_sparsity_coeff {msg}")
        if self.param_match_coeff == 0:
            logger.warning(f"param_match_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert (
                self.lr_exponential_halflife is not None
            ), "lr_exponential_halflife must be set if lr_schedule is exponential"

        if self.full_rank:
            assert not self.unit_norm_matrices, "Can't unit norm matrices if full rank"

        if (
            self.full_rank
            and isinstance(self.task_config, PiecewiseConfig)
            and not self.task_config.handcoded_AB
            and not self.task_config.decompose_bias
        ):
            raise ValueError("Must have one of handcoded_AB or decompose_bias set")

        if (
            not self.full_rank
            and isinstance(self.task_config, PiecewiseConfig)
            and self.task_config.decompose_bias
        ):
            raise ValueError("Cannot decompose bias in rank 1 case")

        if self.topk_param_attrib_coeff is not None and not isinstance(
            self.task_config, PiecewiseConfig
        ):
            raise ValueError("topk_param_attrib_coeff is currenlty only suppported for piecewise")

        if self.distil_from_target and not isinstance(self.task_config, PiecewiseConfig):
            raise ValueError("distil_from_target is currently only supported for piecewise")

        return self


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    if config.pnorm is not None:
        run_suffix += f"p{config.pnorm:.2e}_"
    if config.lp_sparsity_coeff is not None:
        run_suffix += f"lpsp{config.lp_sparsity_coeff:.2e}_"
    if config.orthog_coeff is not None:
        run_suffix += f"orth{config.orthog_coeff:.2e}_"
    if config.topk is not None:
        run_suffix += f"topk{config.topk:.2e}_"
    if config.topk_recon_coeff is not None:
        run_suffix += f"topkrecon{config.topk_recon_coeff:.2e}_"
    if config.topk_l2_coeff is not None:
        run_suffix += f"topkl2_{config.topk_l2_coeff:.2e}_"
    if config.topk_act_recon_coeff is not None:
        run_suffix += f"topkactrecon_{config.topk_act_recon_coeff:.2e}_"
    if config.topk_param_attrib_coeff is not None:
        run_suffix += f"topkattrib_{config.topk_param_attrib_coeff:.2e}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"attr-{config.attribution_type[:3]}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


def get_lr_schedule_fn(
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"],
    lr_exponential_halflife: PositiveFloat | None = None,
) -> Callable[[int, int], float]:
    if lr_schedule == "linear":
        return lambda step, steps: 1 - (step / steps)
    elif lr_schedule == "constant":
        return lambda *_: 1.0
    elif lr_schedule == "cosine":
        return lambda step, steps: 1.0 if steps == 1 else np.cos(0.5 * np.pi * step / (steps - 1))
    elif lr_schedule == "exponential":
        assert lr_exponential_halflife is not None  # Should have been caught by model validator
        halflife = lr_exponential_halflife
        gamma = 0.5 ** (1 / halflife)
        logger.info(f"Using exponential LR schedule with halflife {halflife} steps (gamma {gamma})")
        return lambda step, steps: gamma**step
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")


def get_step_pnorm(step: int, total_steps: int, pnorm_end: float | None = None) -> float:
    if pnorm_end is None:
        return 1.0
    progress = step / total_steps
    return 1 + (pnorm_end - 1) * progress


def get_sparsity_coeff_linear_warmup(
    step: int, steps: int, max_sparsity_coeff: float, sparsity_warmup_pct: float
) -> float:
    warmup_steps = int(steps * sparsity_warmup_pct)
    if step < warmup_steps:
        return max_sparsity_coeff * (step / warmup_steps)
    return max_sparsity_coeff


def get_lr_with_warmup(
    step: int,
    steps: int,
    lr: float,
    lr_schedule_fn: Callable[[int, int], float],
    lr_warmup_pct: float,
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_schedule_fn(step - warmup_steps, steps - warmup_steps)


def calc_recon_mse(
    output: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    labels: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss


def calc_topk_l2_rank_one(
    As_and_Bs_vals: list[tuple[Float[Tensor, "d_layer_in k"], Float[Tensor, "k d_layer_out"]]],
    topk_mask: Bool[Tensor, "batch k"] | Bool[Tensor, "batch n_instances k"],
    n_params: int,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Args:
        all_As_and_Bs: The A and B matrices for each layer.
        topk_mask: The topk mask to use for the L2 penalty.
        n_params: The number of decomposable parameters in the model.
    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    n_instances = topk_mask.shape[1] if topk_mask.ndim == 3 else None
    accumulate_shape = (n_instances,) if n_instances is not None else ()

    topk_l2_penalty = torch.zeros(accumulate_shape, device=As_and_Bs_vals[0][0].device)
    batch_size = topk_mask.shape[0]
    for A, B in As_and_Bs_vals:
        # A: [d_in, k] or [n_instances, d_in, k]
        # B: [k, d_out] or [n_instances, k, d_out]
        # topk_mask: [batch, k] or [batch, n_instances, k]
        A_topk = torch.einsum("...fk,b...k ->b...fk", A, topk_mask)
        AB_topk = torch.einsum("b...fk,...kh->b...fh", A_topk, B)
        topk_l2_penalty = topk_l2_penalty + ((AB_topk) ** 2).sum(dim=(0, -2, -1))

    return topk_l2_penalty / n_params / batch_size


def calc_topk_l2_full_rank(
    subnet_param_vals: list[
        Float[Tensor, "k d_out"]
        | Float[Tensor, "k d_in d_out"]
        | Float[Tensor, "n_instances k d_out"]
        | Float[Tensor, "n_instances k d_in d_out"]
    ],
    topk_mask: Bool[Tensor, "batch k"] | Bool[Tensor, "batch n_instances k"],
    n_params: int,
    n_instances: int | None = None,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        subnetwork_params: The parameters of the subnetwork.
        topk_mask: The topk mask to use for the L2 penalty.
        n_params: The number of decomposable parameters in the model.
        n_instances: The number of instances in the model.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    assert len(subnet_param_vals) > 0, "No subnetwork parameters provided"

    accumulate_shape = (n_instances,) if n_instances is not None else ()

    topk_mask = topk_mask.to(subnet_param_vals[0].dtype)
    topk_l2_penalty = torch.zeros(accumulate_shape, device=subnet_param_vals[0].device)
    batch_size = topk_mask.shape[0]
    for subnetwork_param_val in subnet_param_vals:
        if n_instances is None:
            # subnetwork_param_val: [k, d_in, d_out] or [k, d_out] (if bias param)
            # topk_mask: [batch, k]
            ein_str = "k ... d_out, batch k -> batch ... d_out"
            # mean over all dims
            assert subnetwork_param_val.ndim in (3, 2), "Invalid number of dimensions"
            mean_dims = tuple(range(subnetwork_param_val.ndim))
        else:
            # subnetwork_param_val: [n_instances, k, d_in, d_out] or [n_instances, k, d_out]
            # topk_mask: [batch, n_instances, k]
            ein_str = "n_instances k ... d_out, batch n_instances k -> batch n_instances ... d_out"
            # mean over all dims except the n_instances dim
            assert subnetwork_param_val.ndim in (4, 3), "Invalid number of dimensions"
            mean_dims = (0, -2, -1) if subnetwork_param_val.ndim == 4 else (0, -1)

        topk_params = einops.einsum(subnetwork_param_val, topk_mask, ein_str)
        topk_l2_penalty = topk_l2_penalty + ((topk_params) ** 2).sum(dim=mean_dims)

    return topk_l2_penalty / n_params / batch_size


def calc_param_match_loss(
    pretrained_weights: dict[str, Float[Tensor, "n_instances d_out"] | Float[Tensor, " d_out"]],
    subnetwork_params_summed: dict[
        str, Float[Tensor, "n_instances d_out"] | Float[Tensor, " d_out"]
    ],
    param_map: dict[str, str],
    n_params: int,
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the parameter match loss.

    This is the L2 difference between the combined parameter matrices of the SPDModel and the
    target params.

    Args:
        pretrained_weights: The pretrained weights to be matched. May have an n_instances and/or
            d_in dimension.
        subnetwork_params_summed: The parameters of the SPDModel (that have already been summed over
            the subnetwork dimension). May have an n_instances and/or d_in dimension.
        param_map: A map from keys in pretrained_weights to keys in subnetwork_params_summed.
        has_instance_dim: Whether the model has an n_instances dimension.
        n_params: The number of parameters in the model.

    Returns:
        The parameter match loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    device = next(iter(subnetwork_params_summed.values())).device
    param_match_loss = torch.tensor(0.0, device=device)
    for target_param_name, subnetwork_param_name in param_map.items():
        pretrained_weight = pretrained_weights[target_param_name]
        subnetwork_param = subnetwork_params_summed[subnetwork_param_name]
        if has_instance_dim:
            # params: [n_instances, d_out] or [n_instances, d_in, d_out]
            assert pretrained_weight.ndim in (3, 2)
            mean_dims = (-2, -1) if pretrained_weight.ndim == 3 else (-1,)
        else:
            # params: [d_out] or [d_in, d_out]
            assert pretrained_weight.ndim in (2, 1)
            mean_dims = (-2, -1) if pretrained_weight.ndim == 2 else (-1,)
        param_match_loss = param_match_loss + ((subnetwork_param - pretrained_weight) ** 2).sum(
            dim=mean_dims
        )
    return param_match_loss / n_params


def calc_orthog_loss_full_rank(
    subnet_param_vals: list[
        Float[Tensor, "k d_out"]
        | Float[Tensor, "k d_in d_out"]
        | Float[Tensor, "n_instances k d_out"]
        | Float[Tensor, "n_instances k d_in d_out"]
    ],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the sum of the absolute values of inner products of different subnets.

    NOTE: We could and maybe should try L2 instead of absolute. Note that it is important that we
    use the dot product rather than the cosine sim normalized by layer (though we may normalize
    at the end).

    Args:
        subnetwork_params: The parameters of the SPDModel.
        has_instance_dim: Whether the model has an n_instances dimension.
    Returns:
        The orthogonality loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    first_param = subnet_param_vals[0]
    if has_instance_dim:
        # params: [n_instances, k, d_out] or [n_instances, k, d_in, d_out]
        assert all(param.ndim in (3, 4) for param in subnet_param_vals), "Invalid number of dims"
        k = first_param.shape[1]
        dot_prods = torch.zeros((first_param.shape[0], k, k), device=first_param.device)
        ein_str = "n_instances k1 ... d_out, n_instances k2 ... d_out -> n_instances k1 k2"
    else:
        # params: [k, d_out] or [k, d_in, d_out]
        assert all(param.ndim in (2, 3) for param in subnet_param_vals), "Invalid number of dims"
        k = first_param.shape[0]
        dot_prods = torch.zeros((k, k), device=first_param.device)
        ein_str = "k1 ... d_out, k2 ... d_out -> k1 k2"

    for subnet in subnet_param_vals:
        dot_prods += einops.einsum(subnet, subnet, ein_str)

    # Multiply the k l diagonal by 0
    dot_prods.diagonal(dim1=-2, dim2=-1).zero_()
    orthog_loss = (dot_prods.abs()).mean(dim=(-2, -1))
    return orthog_loss


def calc_lp_sparsity_loss_rank_one(
    out: Float[Tensor, "batch n_instances d_model_out"] | Float[Tensor, "batch d_model_out"],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
    inner_acts: dict[str, Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"]],
    B_params: dict[str, Float[Tensor, "n_instances k d_out"] | Float[Tensor, "k d_out"]],
    step_pnorm: float,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    Unlike the attributions we calculate for topk in `spd.utils.calc_attributions`, in this function
    we calculate the derivative w.r.t. the layer activations and multiply by that layer's B matrix.
    This will give the same gradient as taking the derivative w.r.t. the inner_acts using the chain
    rule, but importantly it puts the B matrix in the computational graph for this calculation so
    backprop can pass through it (autograd.grad will not build a computational graph from
    intermediate tensors
    https://gist.github.com/danbraunai-apollo/388c3c76be92922cf7b2a2f7da7d0d43). This is a
    (somewhat arbitrary) decision to include this layer's B matrix but not future layer parameters
    in the sparsity loss. We don't do this in topk because topk isn't a differentiable operation
    anyway.

    Args:
        out: The output of the model.
        layer_acts: Activations at the output of each layer (i.e. after both A and B transformations).
        inner_acts: The inner acts of the model (i.e. the set of subnetwork activations after the A
            transformation for each parameter matrix).
        B_params: The B matrix of each rank one layer.
        step_pnorm: The pnorm at the current step.

    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    assert layer_acts.keys() == inner_acts.keys() == B_params.keys()
    first_param_name = next(iter(layer_acts.keys()))
    attributions = torch.zeros_like(inner_acts[first_param_name], requires_grad=True)
    for feature_idx in range(out.shape[-1]):
        grad_layer_acts = torch.autograd.grad(
            out[..., feature_idx].sum(),
            list(layer_acts.values()),
            retain_graph=True,
        )
        sparsity_inner = torch.zeros_like(attributions, requires_grad=True)
        for i, param_matrix_name in enumerate(layer_acts.keys()):
            # h_i * grad_h_i
            sparsity_inner = sparsity_inner + (
                inner_acts[param_matrix_name]
                * torch.einsum(
                    "...o,...ko->...k", grad_layer_acts[i].detach(), B_params[param_matrix_name]
                )
            )

        attributions = attributions + sparsity_inner**2
    attributions = attributions / out.shape[-1]

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((attributions.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def calc_lp_sparsity_loss_full_rank(
    out: Float[Tensor, "batch n_instances d_model_out"] | Float[Tensor, "batch d_model_out"],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
    inner_acts: dict[
        str, Float[Tensor, "batch n_instances k d_out"] | Float[Tensor, "batch k d_out"]
    ],
    step_pnorm: float,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    Args:
        out: The output of the model.
        layer_acts: The activations of each layer (after summing over the subnetworks).
        inner_acts: The activations of each subnetwork (before summing over the subnetworks).
        step_pnorm: The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    attributions = calc_grad_attributions_full_rank(out, inner_acts, layer_acts)

    # Average the attributions over the output dimensions
    d_model_out = out.shape[-1]
    attributions = attributions / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((attributions.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def calc_topk_param_attrib_loss(
    target_out: Float[Tensor, "batch n_instances d_model_out"] | Float[Tensor, "batch d_model_out"],
    target_params: dict[str, Tensor],
    subnetwork_params: dict[str, Tensor],
    topk_mask: Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"],
    target_pre_acts: dict[str, Tensor],
    target_post_acts: dict[str, Tensor],
    has_instance_dim: bool,
    n_params: int,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Attribution patch loss of original params to sum of active subnetwork params.

    This function is (an efficient implementation of) dout/dW_target * (W_spd - W_target) where
    W_spd is the sum of currently-active (topk) subnetworks. The actual implementation is based on
    activations to simplify (and maybe speed up) calculations. This is because activations
    (i) have a batch dimension which makes autograd simpler and (ii) only have one not two embedding
    dimensions dimensions.

    Formula:
        d(f(x, W)) / d(a(x, W)) * (W - spd_W) * p(x, W)
    where
    - a(x, W) are the activations after the parameter matrix is applied in the target model,
    - p(x, W) are the activations before the parameter matrix is applied in the SPD model (
        this is set to 1 if W is a bias parameter)
    - f(x, W) is the output of the target model.

    Args:
        target_out: The output of the target model for the batch.
        target_params: The parameters of the target model which are decomposable.
        subnetwork_params: The parameters of the SPD model.
        topk_mask: The topk mask for the SPD model on the batch.
        target_pre_acts: The activations before the parameter matrix is applied in the target
            model.
        target_post_acts: The activations after the parameter matrix is applied in the target
            model.
        has_instance_dim: Whether the model has an n_instances dimension.
        n_params: The number of decomposable parameters in the model.

    Returns:
        The topk parameter attribution loss. Will have an n_instances dimension if the model has an
            n_instances dimension.
    """
    assert target_params.keys() == target_pre_acts.keys() == target_post_acts.keys()
    # Every parameter that we are decomposing must have an entry in target_params
    assert set(target_params.keys()) <= set(subnetwork_params.keys())

    device = next(iter(subnetwork_params.values())).device
    loss = torch.tensor(0.0, device=device)
    for out_idx in range(target_out.shape[-1]):
        # Get the derivative of the output w.r.t. the target_post_acts
        dout_d_post_acts = torch.autograd.grad(
            target_out[..., out_idx].sum(), list(target_post_acts.values()), retain_graph=True
        )
        loss_out_idx = torch.tensor(0.0, device=device)
        for i, param_name in enumerate(subnetwork_params):
            # Sum over the subnetwork dim
            ein_str = "k i ..., b i k -> b i ..." if has_instance_dim else "k ..., b k -> b ..."
            topk_subnet_sum = einops.einsum(
                subnetwork_params[param_name],
                topk_mask.to(dtype=subnetwork_params[param_name].dtype),
                ein_str,
            )
            param_diff = target_params[param_name] - topk_subnet_sum

            if "bias" in param_name:
                # Derivative @ param_diff
                ein_str = (
                    "b i d_out, b i d_out -> b i" if has_instance_dim else "b d_out, b d_out -> b"
                )
                param_loss = einops.einsum(dout_d_post_acts[i].detach(), param_diff, ein_str)
            else:
                # Derivative @ param_diff @ pre_acts
                ein_str = (
                    "b i d_out, b i d_in d_out, b i d_in -> b i"
                    if has_instance_dim
                    else "b d_out, b d_in d_out, b d_in -> b"
                )
                param_loss = einops.einsum(
                    dout_d_post_acts[i].detach(),
                    param_diff,
                    target_pre_acts[param_name],
                    ein_str,
                )

            loss_out_idx = loss_out_idx + param_loss

        loss = loss + loss_out_idx**2
    loss = (loss / target_out.shape[-1] + 1e-16).mean(dim=0)  # Mean over the batch dim
    return loss / n_params


def calc_topk_act_recon(
    target_post_acts: dict[
        str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]
    ],
    layer_acts_topk: dict[
        str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]
    ],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """MSE between all target model activations and the sum of the topk subnetwork activations.

    Args:
        target_post_acts: The activations after each layer in the target model.
        layer_acts_topk: The activations after each subnetwork in the SPD model summed over the topk
            subnetworks.

    Returns:
        The activation reconstruction loss. Will have an n_instances dimension if the model has an
            n_instances dimension, otherwise a scalar.
    """
    assert target_post_acts.keys() == layer_acts_topk.keys(), "Layer keys must match"

    device = next(iter(layer_acts_topk.values())).device

    total_act_dim = 0  # Accumulate the d_out over all layers for normalization
    loss = torch.zeros(1, device=device)
    for layer_name in target_post_acts:
        total_act_dim += target_post_acts[layer_name].shape[-1]

        error = ((target_post_acts[layer_name] - layer_acts_topk[layer_name]) ** 2).sum(dim=-1)
        loss = loss + error

    # Normalize by the total number of output dimensions and mean over the batch dim
    return (loss / total_act_dim).mean(dim=0)


def optimize(
    model: SPDModel | SPDFullRankModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model | None,
    param_map: dict[str, str] | None = None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)

    has_instance_dim = hasattr(model, "n_instances")

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

    if pretrained_model is not None:
        if config.param_match_coeff is not None:
            assert param_map is not None, "Need a param_map for param_match loss"
            # Check that our param_map contains all the decomposable param names
            assert set(param_map.keys()) == set(pretrained_model.all_decomposable_params().keys())
            assert set(param_map.values()) == set(model.all_subnetwork_params_summed().keys())

        pretrained_model.to(device=device)
    n_params = sum(p.numel() for p in list(model.all_subnetwork_params_summed().values()))
    if has_instance_dim:
        # All subnetwork param have an n_instances dimension
        n_params = n_params / model.n_instances

    step_lp_sparsity_coeff = None
    step_topk_recon_coeff = None
    epoch = 0
    total_samples = 0
    data_iter = iter(dataloader)
    for step in tqdm(range(config.steps + 1), ncols=0):
        if config.unit_norm_matrices:
            assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
            model.set_matrices_to_unit_norm()

        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in opt.param_groups:
            group["lr"] = step_lr

        step_pnorm = None

        opt.zero_grad(set_to_none=True)
        try:
            batch, labels = next(data_iter)
        except StopIteration:
            tqdm.write(f"Epoch {epoch} finished, starting new epoch")
            epoch += 1
            data_iter = iter(dataloader)
            batch, labels = next(data_iter)

        batch = batch.to(device=device)
        labels = labels.to(device=device)

        pre_acts = None
        post_acts = None
        if pretrained_model is not None:
            labels, pre_acts, post_acts = pretrained_model(batch)

        total_samples += batch.shape[0]

        if config.topk_recon_coeff is not None:
            step_topk_recon_coeff = get_sparsity_coeff_linear_warmup(
                step=step,
                steps=config.steps,
                max_sparsity_coeff=config.topk_recon_coeff,
                sparsity_warmup_pct=config.sparsity_warmup_pct,
            )
        if config.lp_sparsity_coeff is not None:
            step_lp_sparsity_coeff = get_sparsity_coeff_linear_warmup(
                step=step,
                steps=config.steps,
                max_sparsity_coeff=config.lp_sparsity_coeff,
                sparsity_warmup_pct=config.sparsity_warmup_pct,
            )

        # Do a forward pass with all subnetworks
        out, layer_acts, inner_acts = model(batch)

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, labels, has_instance_dim)

        orthog_loss = None
        if config.orthog_coeff is not None:
            assert config.full_rank, "Orthogonality loss only works in full rank models"
            subnet_param_vals = list(model.all_subnetwork_params().values())
            if config.distil_from_target:
                # Remove the final subnetwork index from all params
                subnet_param_vals = [
                    param[:, :-1] if has_instance_dim else param[:-1] for param in subnet_param_vals
                ]

            orthog_loss = calc_orthog_loss_full_rank(subnet_param_vals=subnet_param_vals)

        param_match_loss = None
        if config.param_match_coeff is not None:
            assert pretrained_model is not None, "Need a pretrained model for param_match loss"
            assert param_map is not None, "Need a param_map for param_match loss"
            param_match_loss = calc_param_match_loss(
                pretrained_weights=pretrained_model.all_decomposable_params(),
                subnetwork_params_summed=model.all_subnetwork_params_summed(),
                param_map=param_map,
                n_params=n_params,
                has_instance_dim=has_instance_dim,
            )

        lp_sparsity_loss = None
        if config.lp_sparsity_coeff is not None:
            step_pnorm = config.pnorm or get_step_pnorm(step, config.steps, config.pnorm_end)
            if config.full_rank:
                lp_sparsity_loss = calc_lp_sparsity_loss_full_rank(
                    out=out, layer_acts=layer_acts, inner_acts=inner_acts, step_pnorm=step_pnorm
                )
            else:
                lp_sparsity_loss = calc_lp_sparsity_loss_rank_one(
                    out=out,
                    layer_acts=layer_acts,
                    inner_acts=inner_acts,
                    B_params={k: tup[1] for k, tup in model.all_As_and_Bs().items()},
                    step_pnorm=step_pnorm,
                )

        (
            out_topk,
            topk_l2_loss,
            topk_recon_loss,
            topk_mask,
            topk_param_attrib_loss,
            topk_act_recon_loss,
        ) = None, None, None, None, None, None
        if config.topk is not None:
            if config.attribution_type == "ablation":
                attribution_scores = calc_ablation_attributions(model=model, batch=batch, out=out)
            elif config.attribution_type == "gradient":
                if config.full_rank:
                    attribution_scores = calc_grad_attributions_full_rank(
                        out=out, inner_acts=inner_acts, layer_acts=layer_acts
                    )
                else:
                    attribution_scores = calc_grad_attributions_rank_one(
                        out=out, inner_acts_vals=list(inner_acts.values())
                    )
            elif config.attribution_type == "activation":
                assert post_acts is not None
                assert config.full_rank, "Activation attributions only supported for full rank"
                attribution_scores = calc_activation_attributions(inner_acts=inner_acts)
            else:
                raise ValueError(f"Invalid attribution type: {config.attribution_type}")

            # We always assume the final subnetwork is the one we want to distil
            topk_attrs = (
                attribution_scores[..., :-1] if config.distil_from_target else attribution_scores
            )
            topk_mask = calc_topk_mask(topk_attrs, config.topk, batch_topk=config.batch_topk)
            if config.distil_from_target:
                # Add back the final subnetwork index to the topk mask and set it to True
                last_subnet_mask = torch.ones(
                    (*topk_mask.shape[:-1], 1), dtype=torch.bool, device=device
                )
                topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1)

            # Do a forward pass with only the topk subnetworks
            out_topk, layer_acts_topk, inner_acts_topk = model(batch, topk_mask=topk_mask)

            if config.topk_l2_coeff is not None:
                if config.full_rank:
                    assert isinstance(model, SPDFullRankModel)
                    topk_l2_loss = calc_topk_l2_full_rank(
                        subnet_param_vals=list(model.all_subnetwork_params().values()),
                        topk_mask=topk_mask,
                        n_params=n_params,
                        n_instances=getattr(model, "n_instances", None),
                    )
                else:
                    topk_l2_loss = calc_topk_l2_rank_one(
                        As_and_Bs_vals=list(model.all_As_and_Bs().values()),
                        topk_mask=topk_mask,
                        n_params=n_params,
                    )

            if config.topk_recon_coeff is not None:
                assert out_topk is not None
                topk_recon_loss = calc_recon_mse(out_topk, labels, has_instance_dim)

            if config.topk_param_attrib_coeff is not None:
                assert pretrained_model is not None, "Need target model for topk_param_attrib_loss"
                assert pre_acts is not None and post_acts is not None
                topk_param_attrib_loss = calc_topk_param_attrib_loss(
                    target_out=labels,
                    target_params=pretrained_model.all_decomposable_params(),
                    subnetwork_params=model.all_subnetwork_params(),
                    topk_mask=topk_mask,
                    target_pre_acts=pre_acts,
                    target_post_acts=post_acts,
                    has_instance_dim=has_instance_dim,
                    n_params=n_params,
                )

            if config.topk_act_recon_coeff is not None:
                assert layer_acts_topk is not None
                assert post_acts is not None
                topk_act_recon_loss = calc_topk_act_recon(
                    target_post_acts=post_acts, layer_acts_topk=layer_acts_topk
                )

        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
        if orthog_loss is not None:
            assert config.orthog_coeff is not None
            loss = loss + config.orthog_coeff * orthog_loss.mean()
        if param_match_loss is not None:
            assert config.param_match_coeff is not None
            loss = loss + config.param_match_coeff * param_match_loss.mean()
        if config.out_recon_coeff is not None:
            loss = loss + config.out_recon_coeff * out_recon_loss.mean()
        if lp_sparsity_loss is not None:
            assert step_lp_sparsity_coeff is not None
            loss = loss + step_lp_sparsity_coeff * lp_sparsity_loss.mean()
        if topk_recon_loss is not None:
            assert step_topk_recon_coeff is not None
            loss = loss + step_topk_recon_coeff * topk_recon_loss.mean()
        if topk_l2_loss is not None:
            assert config.topk_l2_coeff is not None
            loss = loss + config.topk_l2_coeff * topk_l2_loss.mean()
        if topk_param_attrib_loss is not None:
            assert config.topk_param_attrib_coeff is not None
            loss = loss + config.topk_param_attrib_coeff * topk_param_attrib_loss.mean()
        if topk_act_recon_loss is not None:
            assert config.topk_act_recon_coeff is not None
            loss = loss + config.topk_act_recon_coeff * topk_act_recon_loss.mean()

        # Logging
        if step % config.print_freq == 0:
            # If using multiple instances, print the losses as tensors in new lines
            nl = "\n" if has_instance_dim else " "
            tqdm.write(f"Step {step}")
            tqdm.write(f"Total loss: {loss.item()}")
            if step_pnorm is not None:
                tqdm.write(f"Current pnorm:{nl}{step_pnorm}")
            if lp_sparsity_loss is not None:
                tqdm.write(f"LP sparsity loss:{nl}{lp_sparsity_loss}")
            if topk_recon_loss is not None:
                tqdm.write(f"Topk recon loss:{nl}{topk_recon_loss}")
            tqdm.write(f"Out recon loss:{nl}{out_recon_loss}")
            if topk_l2_loss is not None:
                tqdm.write(f"topk l2 loss:{nl}{topk_l2_loss}")
            if param_match_loss is not None:
                tqdm.write(f"Param match loss:{nl}{param_match_loss}")
            if orthog_loss is not None:
                tqdm.write(f"Orthog loss:{nl}{orthog_loss}")
            if topk_param_attrib_loss is not None:
                tqdm.write(f"Topk param attrib loss:{nl}{topk_param_attrib_loss}")
            if topk_act_recon_loss is not None:
                tqdm.write(f"Topk act recon loss:{nl}{topk_act_recon_loss}")
            if config.wandb_project:
                wandb.log(
                    {
                        "pnorm": step_pnorm,
                        "lr": step_lr,
                        "total_loss": loss.mean().item(),
                        "lp_sparsity_loss": lp_sparsity_loss.mean().item()
                        if lp_sparsity_loss is not None
                        else None,
                        "topk_recon_loss": topk_recon_loss.mean().item()
                        if topk_recon_loss is not None
                        else None,
                        "recon_loss": out_recon_loss.mean().item(),
                        "param_match_loss": param_match_loss.mean().item()
                        if param_match_loss is not None
                        else None,
                        "topk_l2_loss": topk_l2_loss.mean().item()
                        if topk_l2_loss is not None
                        else None,
                        "orthog_loss": orthog_loss.mean().item()
                        if orthog_loss is not None
                        else None,
                        "topk_param_attrib_loss": topk_param_attrib_loss.mean().item()
                        if topk_param_attrib_loss is not None
                        else None,
                        "topk_act_recon_loss": topk_act_recon_loss.mean().item()
                        if topk_act_recon_loss is not None
                        else None,
                    },
                    step=step,
                )

        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
        ):
            fig_dict = plot_results_fn(
                model=model,
                target_model=pretrained_model,
                step=step,
                out_dir=out_dir,
                device=device,
                config=config,
                topk_mask=topk_mask,
            )
            if config.wandb_project:
                wandb.log(
                    {k: wandb.Image(v) for k, v in fig_dict.items()},
                    step=step,
                )

        if (
            config.save_freq is not None
            and step % config.save_freq == 0
            and step > 0
            and out_dir is not None
        ):
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            tqdm.write(f"Saved model to {out_dir / f'model_{step}.pth'}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=out_dir)

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward()

            if step % config.print_freq == 0 and config.wandb_project:
                # Calculate gradient norm
                grad_norm: float = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm()  # type: ignore
                    wandb.log({"grad_norm": grad_norm}, step=step)

            if config.unit_norm_matrices:
                assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
                model.fix_normalized_adam_gradients()

            opt.step()
