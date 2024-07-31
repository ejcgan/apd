"""Run SPD on a model."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.models import DeepLinearComponentModel, DeepLinearModel
from spd.types import RootPath
from spd.utils import (
    permute_to_identity,
)

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    topk: int | None = None
    batch_size: int
    steps: int
    print_freq: int
    save_freq: int | None = None
    lr: float
    max_sparsity_coeff: float
    n_features: int | None = None
    n_layers: int | None = None
    n_instances: int | None = None
    k: int | None = None
    pnorm: float | None = None
    pnorm_end: float | None = None
    lr_scale: Literal["linear", "constant", "cosine"] = "constant"
    lr_warmup_pct: float = 0.0
    sparsity_loss_type: Literal["jacobian"] = "jacobian"
    sparsity_warmup_pct: float = 0.0
    pretrained_model_path: RootPath | None = None


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def get_current_pnorm(step: int, total_steps: int, pnorm_end: float | None = None) -> float:
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
    step: int, steps: int, lr: float, lr_scale_fn: Callable[[int, int], float], lr_warmup_pct: float
) -> float:
    warmup_steps = int(steps * lr_warmup_pct)
    if step < warmup_steps:
        return lr * (step / warmup_steps)
    return lr * lr_scale_fn(step - warmup_steps, steps - warmup_steps)


def plot_inner_acts(
    batch: Float[Tensor, "batch n_instances n_features"],
    inner_acts: list[Float[Tensor, "batch n_instances k"]],
) -> plt.Figure:
    """Plot the inner acts for the first batch_elements in the batch.

    The first row is the raw batch information, the following rows are the inner acts per layer.
    """
    n_layers = len(inner_acts)
    n_instances = batch.shape[1]

    fig, axs = plt.subplots(
        n_layers + 1,
        n_instances,
        figsize=(2.5 * n_instances, 2.5 * (n_layers + 1)),
        squeeze=False,
        sharey=True,
    )

    cmap = "Blues"
    # Add the batch data
    for i in range(n_instances):
        ax = axs[0, i]
        data = batch[:, i, :].detach().cpu().float().numpy()
        ax.matshow(data, vmin=0, vmax=np.max(data), cmap=cmap)

        ax.set_title(f"Instance {i}")
        if i == 0:
            ax.set_ylabel("Inputs")
        elif i == n_instances - 1:
            ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
            ax.yaxis.set_label_position("right")

        # Set an xlabel for each plot
        ax.set_xlabel("n_features")

        ax.set_xticks([])
        ax.set_yticks([])

    # Add the inner acts
    for layer in range(n_layers):
        for i in range(n_instances):
            ax = axs[layer + 1, i]
            instance_data = inner_acts[layer][:, i, :].abs().detach().cpu().float().numpy()
            ax.matshow(instance_data, vmin=0, vmax=np.max(instance_data), cmap=cmap)

            if i == 0:
                ax.set_ylabel(f"h_{layer}")
            elif i == n_instances - 1:
                ax.set_ylabel("batch_idx", rotation=-90, va="bottom", labelpad=15)
                ax.yaxis.set_label_position("right")

            if layer == n_layers - 1:
                ax.set_xlabel("k")

            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    return fig


def collect_inner_act_data(
    model: DeepLinearComponentModel, device: str, topk: int | None = None
) -> tuple[
    Float[Tensor, "batch n_instances n_features"], list[Float[Tensor, "batch n_instances k"]]
]:
    """
    Collect inner activation data for visualization.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the inner activations. It then permutes the activations to align with the identity.

    Args:
        model (DeepLinearComponentModel): The model to collect data from.
        device (str): The device to run computations on.
        topk (int, optional): The number of topk values to use in each layer. If None, use all
            activations.

    Returns:
        - The input test batch (identity matrix expanded over instance dimension).
        - A list of permuted inner activations for each layer.

    """
    test_batch = einops.repeat(
        torch.eye(model.n_features, device=device),
        "b f -> b i f",
        i=model.n_instances,
    )

    if topk is not None:
        _, _, test_inner_acts = model.forward_topk(test_batch, topk=topk)
    else:
        _, _, test_inner_acts = model(test_batch)

    test_inner_acts_permuted = []
    for layer in range(model.n_layers):
        test_inner_acts_layer_permuted = []
        for i in range(model.n_instances):
            test_inner_acts_layer_permuted.append(
                permute_to_identity(test_inner_acts[layer][:, i, :].abs())
            )
        test_inner_acts_permuted.append(torch.stack(test_inner_acts_layer_permuted, dim=1))

    return test_batch, test_inner_acts_permuted


def optimize(
    dlc_model: DeepLinearComponentModel,
    config: Config,
    out_dir: Path,
    device: str,
    pretrained_model_path: RootPath | None = None,
) -> None:
    assert (
        (config.pnorm is None and config.pnorm_end is not None)
        or (config.pnorm is not None and config.pnorm_end is None)
        or config.topk is not None
    ), "Exactly one of pnorm and pnorm_end must be set"

    pretrained_weights: list[torch.Tensor] | None = None
    if pretrained_model_path:
        pretrained_model = DeepLinearModel.from_pretrained(pretrained_model_path).to(device)
        pretrained_weights = [weight for weight in pretrained_model.layers]
        assert len(pretrained_weights) == dlc_model.n_layers
        for weight in pretrained_weights:
            weight.requires_grad = False

    opt = torch.optim.AdamW(dlc_model.parameters(), lr=config.lr)

    lr_scale_fn: Callable[[int, int], float]
    if config.lr_scale == "linear":
        lr_scale_fn = linear_lr
    elif config.lr_scale == "constant":
        lr_scale_fn = constant_lr
    elif config.lr_scale == "cosine":
        lr_scale_fn = cosine_decay_lr
    else:
        lr_scale_fn = constant_lr

    total_samples = 0

    for step in tqdm(range(config.steps)):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_scale_fn=lr_scale_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

        current_pnorm = (
            get_current_pnorm(step, config.steps, config.pnorm_end)
            if config.pnorm is None
            else config.pnorm
        )

        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = dlc_model.generate_batch(config.batch_size).to(dtype=torch.float32, device=device)

        total_samples += batch.shape[0]  # don't include the number of instances

        sparsity_coeff = get_sparsity_coeff_linear_warmup(
            step=step,
            steps=config.steps,
            max_sparsity_coeff=config.max_sparsity_coeff,
            sparsity_warmup_pct=config.sparsity_warmup_pct,
        )

        dlc_out_topk = None
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if config.topk is not None:
                # First do a full forward pass and get the gradients w.r.t. inner_acts
                # Stage 1: Do a full forward pass and get the gradients w.r.t inner_acts
                dlc_out, _, inner_acts = dlc_model(batch)
                all_grads = [torch.zeros_like(inner_acts[i]) for i in range(dlc_model.n_layers)]
                for feature_idx in range(dlc_out.shape[-1]):
                    grads = torch.autograd.grad(
                        dlc_out[:, :, feature_idx].sum(), inner_acts, retain_graph=True
                    )
                    for layer_idx in range(dlc_model.n_layers):
                        all_grads[layer_idx] += grads[layer_idx]

                # Now do a full forward pass with topk
                dlc_out_topk, layer_acts, inner_acts_topk = dlc_model.forward_topk(
                    batch, config.topk, all_grads
                )
                assert len(inner_acts_topk) == dlc_model.n_layers
            else:
                dlc_out, layer_acts, inner_acts = dlc_model(batch)

            assert len(inner_acts) == dlc_model.n_layers

            param_match_loss = torch.zeros(dlc_model.n_instances, device=device)
            if pretrained_model_path:
                # If the user passed a pretrained model, then calculate the param_match_loss
                assert pretrained_weights is not None
                for i in range(dlc_model.n_layers):
                    normed_A = dlc_model.layers[i].A / dlc_model.layers[i].A.norm(
                        p=2, dim=-2, keepdim=True
                    )
                    AB = torch.einsum("ifk,ikg->ifg", normed_A, dlc_model.layers[i].B)
                    param_match_loss = param_match_loss + ((AB - pretrained_weights[i]) ** 2).sum(
                        dim=(-2, -1)
                    )

            output_error = (dlc_out - batch) ** 2
            out_recon_loss = einops.reduce(output_error, "b i f -> i", "mean")

            if config.topk is None:
                all_Bs = [dlc_model.layers[i].B for i in range(dlc_model.n_layers)]

                sparsity_loss = torch.zeros_like(layer_acts[0], requires_grad=True)
                for feature_idx in range(dlc_out.shape[-1]):
                    grad_layer_acts = torch.autograd.grad(
                        dlc_out[:, :, feature_idx].sum(),
                        layer_acts,
                        retain_graph=True,
                    )
                    sparsity_inner = torch.zeros_like(sparsity_loss, requires_grad=True)
                    for layer_idx in range(dlc_model.n_layers):
                        # h_i * grad_h_i
                        sparsity_inner = sparsity_inner + (
                            inner_acts[layer_idx]
                            * torch.einsum(
                                "...ih,ikh->...ik",
                                grad_layer_acts[layer_idx].detach(),
                                all_Bs[layer_idx],
                            )
                        )

                    sparsity_loss = sparsity_loss + sparsity_inner**2
                sparsity_loss = sparsity_loss / dlc_out.shape[-1] + 1e-16

                # Note the current_pnorm * 0.5 is because we have the squares of the sparsity inner
                # above
                sparsity_loss = einops.reduce(
                    ((sparsity_loss.abs() + 1e-16) ** (current_pnorm * 0.5)).sum(dim=-1),
                    "b i -> i",
                    "mean",
                )
            else:
                # Assert that dlc_out_topk is not unbound
                assert dlc_out_topk is not None
                output_topk_error = (dlc_out_topk - batch) ** 2
                sparsity_loss = einops.reduce(output_topk_error, "b i f -> i", "mean")

            with torch.inference_mode():
                if step % config.print_freq == config.print_freq - 1 or step == 0:
                    tqdm.write(f"Step {step}")
                    tqdm.write(f"Current pnorm: {current_pnorm}")
                    tqdm.write(f"Sparsity loss: \n{sparsity_loss}")
                    tqdm.write(f"Reconstruction loss: \n{out_recon_loss}")
                    if pretrained_model_path:
                        # param_match_repr = [f"{x:.4f}" for x in param_match_loss]
                        param_match_repr = [f"{x}" for x in param_match_loss]
                        tqdm.write(f"Param match loss: \n{param_match_repr}")

                    test_batch, test_inner_acts = collect_inner_act_data(
                        dlc_model, device, config.topk
                    )

                    fig = plot_inner_acts(batch=test_batch, inner_acts=test_inner_acts)
                    fig.savefig(out_dir / f"inner_acts_{step}.png")
                    plt.close(fig)
                    tqdm.write(f"Saved inner_acts to {out_dir / f'inner_acts_{step}.png'}")

                    if config.wandb_project:
                        wandb.log(
                            {
                                "step": step,
                                "current_pnorm": current_pnorm,
                                "current_lr": step_lr,
                                "sparsity_loss": sparsity_loss.mean().item(),
                                "recon_loss": out_recon_loss.mean().item(),
                                "param_match_loss": param_match_loss.mean().item(),
                                "inner_acts": wandb.Image(fig),
                            },
                            step=step,
                        )

                if config.save_freq is not None and step % config.save_freq == config.save_freq - 1:
                    torch.save(dlc_model.state_dict(), out_dir / f"model_{step}.pth")
                    tqdm.write(f"Saved model to {out_dir / f'model_{step}.pth'}")

            out_recon_loss = out_recon_loss.mean()
            sparsity_loss = sparsity_loss.mean()
            param_match_loss = param_match_loss.mean()

            if pretrained_model_path:
                loss = param_match_loss + sparsity_coeff * sparsity_loss
            else:
                loss = out_recon_loss + sparsity_coeff * sparsity_loss

        loss.backward()
        opt.step()

    torch.save(dlc_model.state_dict(), out_dir / f"model_{config.steps}.pth")
    logger.info(f"Saved model to {out_dir / f'model_{config.steps}.pth'}")
    if config.wandb_project:
        wandb.save(str(out_dir / f"model_{config.steps}.pth"))
