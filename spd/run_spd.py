"""Run SPD on a model."""

import json
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
from spd.utils import calc_attributions_full_rank, calc_attributions_rank_one, calc_topk_mask


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
    simple_bias: bool = False
    handcoded_AB: bool = False


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
    out_recon_coeff: NonNegativeFloat | None = None
    param_match_coeff: NonNegativeFloat | None = 1.0
    topk_recon_coeff: NonNegativeFloat | None = None
    topk_l2_coeff: NonNegativeFloat | None = None
    lp_sparsity_coeff: NonNegativeFloat | None = None
    pnorm: PositiveFloat | None = None
    pnorm_end: PositiveFloat | None = None
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    lr_exponential_halflife: PositiveFloat | None = None
    lr_warmup_pct: Probability = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    sparsity_warmup_pct: Probability = 0.0
    unit_norm_matrices: bool = True
    task_config: DeepLinearConfig | PiecewiseConfig | TMSConfig = Field(
        ..., discriminator="task_name"
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check valid combinations of topk and batch_size
        if self.topk is not None:
            if self.batch_topk:
                if not (self.batch_size * self.topk).is_integer():
                    raise ValueError("batch_size * topk must be an integer when using batch_topk")
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
        return self


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
    output: Float[Tensor, "... n_features"],
    labels: Float[Tensor, "... n_features"],
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
    layer_in_params: list[Float[Tensor, " ... d_in k"]],
    layer_out_params: list[Float[Tensor, " ... k d_out"]],
    topk_mask: Bool[Tensor, "batch ... k"],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        layer_in_params (list[Float[Tensor, " ... d_in k"]]): The input parameters of each layer.
        layer_out_params (list[Float[Tensor, " ... k d_out"]]): The output parameters of each layer.
        topk_mask (Bool[Tensor, "batch ... k"]): The topk mask to use for the L2 penalty.
            Will contain an n_instances dimension if the model has an n_instances dimension.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    batch_size = topk_mask.shape[0]
    n_instances = topk_mask.shape[1] if topk_mask.ndim == 3 else None
    accumulate_shape = (batch_size,) if n_instances is None else (batch_size, n_instances)

    topk_l2_penalty = torch.zeros(accumulate_shape, device=layer_in_params[0].device)
    for A, B in zip(layer_in_params, layer_out_params, strict=True):
        # A: [d_in, k] or [n_instances, d_in, k]
        # B: [k, d_in] or [n_instances, k, d_in]
        # topk_mask: [batch, k] or [batch, n_instances, k]
        A_topk = torch.einsum("...fk,b...k ->b...fk", A, topk_mask)
        AB_topk = torch.einsum("b...fk,...kh->b...fh", A_topk, B)
        topk_l2_penalty = topk_l2_penalty + ((AB_topk) ** 2).mean(dim=(-2, -1))
    # Mean over batch_dim and divide by number of parameter matrices we iterated over
    return topk_l2_penalty.mean(dim=0) / len(layer_in_params)


def calc_topk_l2_full_rank(
    subnetwork_params: list[Float[Tensor, " ... k d_in d_out"]],
    topk_mask: Bool[Tensor, "batch ... k"],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the L2 of the sum of the topk subnetworks.

    Note that we explicitly write the batch dimension to aid understanding. The einsums
    produce the same operation without it. The ... indicates an optional n_instances dimension.

    Args:
        subnetwork_params (list[Float[Tensor, " ... k d_in d_out"]]): The parameters of the
            subnetworks.
        topk_mask (Bool[Tensor, "batch ... k"]): The topk mask to use for the L2 penalty.
            Will contain an n_instances dimension if the model has an n_instances dimension.

    Returns:
        The L2 penalty for the topk subnetworks. One value for each n_instance (used in tms and
            deep linear toy models).
    """
    batch_size = topk_mask.shape[0]
    n_instances = topk_mask.shape[1] if topk_mask.ndim == 3 else None
    accumulate_shape = (batch_size,) if n_instances is None else (batch_size, n_instances)

    topk_l2_penalty = torch.zeros(accumulate_shape, device=subnetwork_params[0].device)
    for subnetwork_param in subnetwork_params:
        # subnetwork_param: [k, d_in, d_out] or [n_instances, k, d_in, d_out]
        # topk_mask: [batch, k] or [batch, n_instances, k]
        topk_mask = topk_mask.float()
        topk_params = einops.einsum(
            subnetwork_param, topk_mask, "... k d_in d_out, batch ... k -> batch ... d_in d_out"
        )
        topk_l2_penalty = topk_l2_penalty + ((topk_params) ** 2).mean(dim=(-2, -1))
    # Mean over batch_dim and divide by number of parameter matrices we iterated over
    return topk_l2_penalty.mean(dim=0) / len(subnetwork_params)


def calc_param_match_loss_rank_one(
    pretrained_weights: list[Float[Tensor, " ... d_in d_out"]],
    layer_in_params: list[Float[Tensor, " ... d_in k"]],
    layer_out_params: list[Float[Tensor, " ... k d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the parameter match loss.

    This is the L2 difference between the AB matrices of the SPDModel and the pretrained weights.

    Args:
        pretrained_weights (list[Float[Tensor, " ... d_in d_out"]]): The pretrained weights to be
            matched.
        layer_in_params (list[Float[Tensor, " ... d_in k"]]): The input parameters of each layer.
        layer_out_params (list[Float[Tensor, " ... k d_out"]]): The output parameters of each layer.

    Returns:
        The parameter match loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    param_match_loss = torch.tensor(0.0, device=layer_in_params[0].device)
    for i, (A, B) in enumerate(zip(layer_in_params, layer_out_params, strict=True)):
        AB = torch.einsum("...fk,...kg->...fg", A, B)
        param_match_loss = param_match_loss + ((AB - pretrained_weights[i]) ** 2).mean(dim=(-2, -1))
    return param_match_loss / len(layer_in_params)


def calc_param_match_loss_full_rank(
    pretrained_weights: list[Float[Tensor, " ... d_in d_out"]],
    subnetwork_params: list[Float[Tensor, " ... k d_in d_out"]],
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the parameter match loss.

    This is the L2 difference between the sum of the subnetwork matrices of the SPDModel and the
    pretrained weights.

    Args:
        pretrained_weights (list[Float[Tensor, " ... d_in d_out"]]): The pretrained weights to be
            matched.
        subnetwork_params (list[Float[Tensor, " ... k d_in d_out"]]): The parameters of the SPDModel

    Returns:
        The parameter match loss of shape [n_instances] if the model has an n_instances dimension,
        otherwise of shape [].
    """
    param_match_loss = torch.tensor(0.0, device=subnetwork_params[0].device)
    for pretrained_weight, subnetwork_param in zip(
        pretrained_weights, subnetwork_params, strict=False
    ):
        summed_param = einops.einsum(subnetwork_param, "... k d_in d_out -> ... d_in d_out")
        param_match_loss = param_match_loss + ((summed_param - pretrained_weight) ** 2).mean(
            dim=(-2, -1)
        )
    return param_match_loss / len(subnetwork_params)


def calc_lp_sparsity_loss_rank_one(
    out: Float[Tensor, "... d_model_out"],
    layer_acts: list[Float[Tensor, "... d_in"]],
    inner_acts: list[Float[Tensor, "... d_in"]],
    layer_out_params: list[Float[Tensor, "... k d_out"]],
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
        out (Float[Tensor, "... d_model_out"]): The output of the model.
        layer_acts (list[Float[Tensor, "... d_in"]]): Activations at the output of each layer (i.e.
            after both A and B transformations).
        inner_acts (list[Float[Tensor, "... d_in"]]): The inner acts of the model (i.e.
            the set of subnetwork activations after the A transformation for each parameter matrix).
        layer_out_params (list[Float[Tensor, "... k d_out"]]): The output parameters of each layer.
        step_pnorm (float): The pnorm at the current step.

    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    assert len(layer_acts) == len(inner_acts) == len(layer_out_params)
    lp_sparsity_loss = torch.zeros_like(inner_acts[0], requires_grad=True)
    for feature_idx in range(out.shape[-1]):
        grad_layer_acts = torch.autograd.grad(
            out[..., feature_idx].sum(),
            layer_acts,
            retain_graph=True,
        )
        sparsity_inner = torch.zeros_like(lp_sparsity_loss, requires_grad=True)
        for param_matrix_idx in range(len(layer_out_params)):
            # h_i * grad_h_i
            sparsity_inner = sparsity_inner + (
                inner_acts[param_matrix_idx]
                * torch.einsum(
                    "...o,...ko->...k",
                    grad_layer_acts[param_matrix_idx].detach(),
                    layer_out_params[param_matrix_idx],
                )
            )

        lp_sparsity_loss = lp_sparsity_loss + sparsity_inner**2
    lp_sparsity_loss = lp_sparsity_loss / out.shape[-1]

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((lp_sparsity_loss.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def calc_lp_sparsity_loss_full_rank(
    out: Float[Tensor, "... d_model_out"],
    layer_acts: list[Float[Tensor, "... d_out"]],
    inner_acts: list[Float[Tensor, "... k d_out"]],
    step_pnorm: float,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    """Calculate the Lp sparsity loss on the attributions (inner_acts * d(out)/d(inner_acts).

    The attributions here now work the same as in the topk case, because there is no B matrix.

    Args:
        out (Float[Tensor, "... d_model_out"]): The output of the model.
        layer_acts (list[Float[Tensor, "... d_out"]]): The activations of each layer.
        inner_acts (list[Float[Tensor, "... k d_out"]]): The activations of each subnetwork.
        step_pnorm (float): The pnorm to use for the sparsity loss.
    Returns:
        The Lp sparsity loss. Will have an n_instances dimension if the model has an n_instances
            dimension.
    """
    n_param_matrices = len(layer_acts)
    assert n_param_matrices == len(inner_acts)
    batch = out.shape[0]
    assert batch == layer_acts[0].shape[0] == inner_acts[0].shape[0]
    lp_sparsity_loss = torch.zeros(inner_acts[0].shape[:-1], requires_grad=True)
    for feature_idx in range(out.shape[-1]):
        grad_layer_acts = torch.autograd.grad(
            out[..., feature_idx].sum(),
            layer_acts,
            retain_graph=True,
        )
        sparsity_inner = torch.zeros_like(lp_sparsity_loss, requires_grad=True)
        for param_matrix_idx in range(n_param_matrices):
            # h_i * grad_h_i
            sparsity_inner += einops.einsum(
                grad_layer_acts[param_matrix_idx].detach(),
                inner_acts[param_matrix_idx],
                "... d_out ,... k d_out -> ... k",
            )

        lp_sparsity_loss = lp_sparsity_loss + sparsity_inner**2
    d_model_out = out.shape[-1]
    lp_sparsity_loss = lp_sparsity_loss / d_model_out

    # step_pnorm * 0.5 is because we have the squares of sparsity_inner terms above
    lp_sparsity_loss = ((lp_sparsity_loss.abs() + 1e-16) ** (step_pnorm * 0.5)).sum(dim=-1)
    lp_sparsity_loss = lp_sparsity_loss.mean(dim=0)  # Mean over batch dim
    return lp_sparsity_loss


def optimize(
    model: SPDModel | SPDFullRankModel,
    config: Config,
    device: str,
    dataloader: DataLoader[tuple[Float[Tensor, "... n_features"], Float[Tensor, "... n_features"]]],
    pretrained_model: Model | None,
    plot_results_fn: Callable[..., dict[str, plt.Figure]] | None = None,
    out_dir: Path | None = None,
) -> None:
    model.to(device=device)

    has_instance_dim = hasattr(model, "n_instances")

    # Note that we expect weight decay to be problematic for spd
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)

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

        if pretrained_model is not None:
            pretrained_model.requires_grad_(False)
            pretrained_model.to(device=device)
            with torch.inference_mode():
                labels = pretrained_model(batch)

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
        assert len(inner_acts) == model.n_param_matrices

        # Calculate losses
        out_recon_loss = calc_recon_mse(out, labels, has_instance_dim)

        param_match_loss = None
        if config.param_match_coeff is not None:
            assert pretrained_model is not None, "Need a pretrained model for param_match loss"
            pretrained_weights = pretrained_model.all_decomposable_params()
            if config.full_rank:
                assert isinstance(model, SPDFullRankModel)
                param_match_loss = calc_param_match_loss_full_rank(
                    pretrained_weights, model.all_subnetwork_params()
                )
            else:
                assert isinstance(model, SPDModel)
                param_match_loss = calc_param_match_loss_rank_one(
                    pretrained_weights, model.all_As(), model.all_Bs()
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
                    layer_out_params=model.all_Bs(),
                    step_pnorm=step_pnorm,
                )

        out_topk, topk_l2_loss, topk_recon_loss = None, None, None
        if config.topk is not None:
            if config.full_rank:
                attribution_scores = calc_attributions_full_rank(
                    out=out,
                    inner_acts=inner_acts,
                    layer_acts=layer_acts,
                )
            else:
                attribution_scores = calc_attributions_rank_one(out=out, inner_acts=inner_acts)

            topk_mask = calc_topk_mask(
                attribution_scores, config.topk, batch_topk=config.batch_topk
            )

            # Do a forward pass with only the topk subnetworks
            out_topk, _, inner_acts_topk = model.forward_topk(batch, topk_mask=topk_mask)
            assert len(inner_acts_topk) == model.n_param_matrices

            if config.topk_l2_coeff is not None:
                if config.full_rank:
                    topk_l2_loss = calc_topk_l2_full_rank(
                        subnetwork_params=model.all_subnetwork_params(), topk_mask=topk_mask
                    )
                else:
                    topk_l2_loss = calc_topk_l2_rank_one(
                        layer_in_params=model.all_As(),
                        layer_out_params=model.all_Bs(),
                        topk_mask=topk_mask,
                    )

            if config.topk_recon_coeff is not None:
                assert out_topk is not None
                topk_recon_loss = calc_recon_mse(out_topk, labels, has_instance_dim)

        # Add up the loss terms
        loss = torch.tensor(0.0, device=device)
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
            if config.wandb_project:
                wandb.log(
                    {
                        "step": step,
                        "pnorm": step_pnorm,
                        "lr": step_lr,
                        "total_loss": loss.mean().item(),
                        "lp_sparsity_coeff": step_lp_sparsity_coeff,
                        "topk_recon_coeff": step_topk_recon_coeff,
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
                    },
                    step=step,
                )

        if (
            plot_results_fn is not None
            and config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or not config.slow_images)
        ):
            fig_dict = plot_results_fn(
                model=model,
                step=step,
                out_dir=out_dir,
                device=device,
                topk=config.topk,
                batch_topk=config.batch_topk,
                slow_images=config.slow_images,
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
            with open(out_dir / "config.json", "w") as f:
                json.dump(config.model_dump(), f, indent=4)
            tqdm.write(f"Saved config to {out_dir / 'config.json'}")

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            loss.backward()
            if config.unit_norm_matrices:
                assert isinstance(model, SPDModel), "Can only norm matrices in SPDModel instances"
                model.fix_normalized_adam_gradients()
            opt.step()
