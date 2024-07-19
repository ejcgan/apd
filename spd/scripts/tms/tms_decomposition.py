"""TMS decomposition script.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from spd.log import logger
from spd.types import RootPath
from spd.utils import (
    calculate_closeness_to_identity,
    init_wandb,
    load_config,
    permute_to_identity,
    set_seed,
)

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
    n_features: int
    n_hidden: int
    n_instances: int
    batch_size: int
    steps: int
    print_freq: int
    lr: float
    max_sparsity_coeff: float
    k: int | None = None
    pnorm: float | None = None
    pnorm_end: float | None = None
    lr_scale: Literal["linear", "constant", "cosine"] = "constant"
    lr_warmup_pct: float = 0.0
    sparsity_loss_type: Literal["jacobian", "dotted"] = "jacobian"
    sparsity_warmup_pct: float = 0.0
    bias_val: float = 0.0
    train_bias: bool = False
    feature_probability: float = 0.05
    pretrained_model_path: RootPath | None = None


class Model(nn.Module):
    def __init__(self, config: Config, device: str = "cuda"):
        super().__init__()
        self.config = config

        k = config.k if config.k is not None else config.n_features

        self.A = nn.Parameter(
            torch.empty((config.n_instances, config.n_features, k), device=device),
        )
        self.B = nn.Parameter(torch.empty((config.n_instances, k, config.n_hidden), device=device))

        bias_data = (
            torch.zeros((config.n_instances, config.n_features), device=device) + config.bias_val
        )
        self.b_final = nn.Parameter(bias_data) if config.train_bias else bias_data

        nn.init.xavier_normal_(self.A)
        # Fix the first instance to the identity to compare losses
        assert (
            config.n_features == k
        ), "Currently only supports n_features == k if fixing first instance to identity"
        self.A.data[0] = torch.eye(config.n_features, device=device)
        nn.init.xavier_normal_(self.B)

        self.feature_probability = config.feature_probability
        self.importance = torch.ones((), device=device)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_0 = torch.einsum("...if,ifk->...ik", features, self.A)
        hidden = torch.einsum("...ik,ikh->...ih", h_0, self.B)

        h_1 = torch.einsum("...ih,ikh->...ik", hidden, self.B)
        hidden_2 = torch.einsum("...ik,ifk->...if", h_1, self.A)

        pre_relu = hidden_2 + self.b_final
        out = F.relu(pre_relu)
        return out, h_0, h_1, hidden, pre_relu

    def generate_batch(self, n_batch: int) -> torch.Tensor:
        feat = torch.rand(
            (n_batch, self.config.n_instances, self.config.n_features), device=self.A.device
        )

        batch = torch.where(
            torch.rand(
                (n_batch, self.config.n_instances, self.config.n_features), device=self.A.device
            )
            <= self.feature_probability,
            feat,
            torch.zeros((), device=self.A.device),
        )
        return batch


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


def plot_A_matrix(x: torch.Tensor, pos_only: bool = False) -> plt.Figure:
    n_instances = x.shape[0]

    fig, axs = plt.subplots(
        1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
    )

    max_abs_val = x.abs().max()
    vmin = -max_abs_val if not pos_only else 0
    vmax = max_abs_val
    cmap = "Blues" if pos_only else "RdBu"
    im = None
    for i in range(n_instances):
        ax = axs[0, i]
        im = ax.matshow(
            x[i, :, :].T.detach().cpu().float().numpy(), vmin=vmin, vmax=vmax, cmap=cmap
        )
        ax.xaxis.set_ticks_position("bottom")
        if i == 0:
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
        else:
            ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("n_features")

    assert im is not None

    plt.subplots_adjust(wspace=0.1, bottom=0.15, top=0.9)
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes((0.3, 0.05, 0.4, 0.02))
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    return fig


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


def optimize(
    model: Model,
    config: Config,
    out_dir: Path,
    device: str,
    pretrained_model_path: RootPath | None = None,
) -> None:
    assert (config.pnorm is None and config.pnorm_end is not None) or (
        config.pnorm is not None and config.pnorm_end is None
    ), "Exactly one of pnorm and pnorm_end must be set"
    assert (
        config.pnorm_end is not None or config.pnorm is not None
    ), "pnorm_end must be set if pnorm is not set"

    pretrained_W = None
    if pretrained_model_path:
        pretrained_W = torch.load(pretrained_model_path)["W"].to(device)
    opt = torch.optim.AdamW(list(model.parameters()), lr=config.lr)

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
        batch = model.generate_batch(config.batch_size)

        total_samples += batch.shape[0]  # don't include the number of instances

        sparsity_coeff = get_sparsity_coeff_linear_warmup(
            step=step,
            steps=config.steps,
            max_sparsity_coeff=config.max_sparsity_coeff,
            sparsity_warmup_pct=config.sparsity_warmup_pct,
        )

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out, h_0, h_1, hidden, pre_relu = model(batch)

            param_match_loss = torch.zeros(model.config.n_instances, device=device)
            if pretrained_model_path:
                # If the user passed a pretrained model, then calculate the param_match_loss
                # Get the Frobenius norm between the pretrained weight and the current model's W
                assert pretrained_W is not None
                param_match_loss = (
                    ((pretrained_W[: model.config.n_instances] - model.A @ model.B) ** 2)
                    .sum(dim=(-2, -1))
                    .sqrt()
                )

            error = model.importance * (batch - out) ** 2
            recon_loss = einops.reduce(error, "b i f -> i", "mean")

            # Note that we want the weights of A and B to update based on the gradient of the loss
            # w.r.t h_0 and h_1. So we can't just calculate the gradient w.r.t these terms directly
            # and then detach.
            if config.sparsity_loss_type == "dotted":
                out_dotted = model.importance * torch.einsum("bih,bih->bi", out, out).sum()
                grad_hidden, grad_pre_relu = torch.autograd.grad(
                    out_dotted, (hidden, pre_relu), create_graph=True
                )
                grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)
                sparsity_loss = h_0 * grad_h_0 + h_1 * grad_h_1
            elif config.sparsity_loss_type == "jacobian":
                sparsity_loss = torch.zeros_like(h_0, requires_grad=True)
                for feature_idx in range(out.shape[-1]):
                    grad_hidden, grad_pre_relu = torch.autograd.grad(
                        out[:, :, feature_idx].sum(),
                        (hidden, pre_relu),
                        grad_outputs=torch.tensor(1.0, device=out.device),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                    grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)

                    sparsity_inner = grad_h_0 * h_0 + grad_h_1 * h_1

                    sparsity_loss = sparsity_loss + sparsity_inner**2
                sparsity_loss = (sparsity_loss / out.shape[-1] + 1e-16).sqrt()
            else:
                raise ValueError(f"Unknown sparsity loss type: {config.sparsity_loss_type}")

            sparsity_loss = einops.reduce(
                ((sparsity_loss.abs() + 1e-16) ** current_pnorm).sum(dim=-1), "b i -> i", "mean"
            )

            with torch.inference_mode():
                if step % config.print_freq == config.print_freq - 1 or step == 0:
                    sparsity_repr = [f"{x:.4f}" for x in sparsity_loss]
                    recon_repr = [f"{x:.4f}" for x in recon_loss]
                    tqdm.write(f"Step {step}")
                    tqdm.write(f"Current pnorm: {current_pnorm}")
                    tqdm.write(f"Sparsity loss: \n{sparsity_repr}")
                    tqdm.write(f"Reconstruction loss: \n{recon_repr}")
                    if pretrained_model_path:
                        param_match_repr = [f"{x:.4f}" for x in param_match_loss]
                        tqdm.write(f"Param match loss: \n{param_match_repr}")

                    closeness_vals: list[float] = []
                    permuted_A_T_list: list[torch.Tensor] = []
                    for i in range(model.config.n_instances):
                        permuted_matrix = permute_to_identity(model.A[i].T.abs())
                        closeness = calculate_closeness_to_identity(permuted_matrix)
                        closeness_vals.append(closeness)
                        permuted_A_T_list.append(permuted_matrix)
                    permuted_A_T = torch.stack(permuted_A_T_list, dim=0)

                    fig = plot_A_matrix(permuted_A_T, pos_only=True)

                    fig.savefig(out_dir / f"A_{step}.png")
                    plt.close(fig)
                    tqdm.write(f"Saved A matrix to {out_dir / f'A_{step}.png'}")
                    if config.wandb_project:
                        wandb.log(
                            {
                                "step": step,
                                "lr": step_lr,
                                "current_pnorm": current_pnorm,
                                "sparsity_loss": sparsity_loss[1:].mean().item(),
                                "recon_loss": recon_loss[1:].mean().item(),
                                "param_match_loss": param_match_loss[1:].mean().item(),
                                "closeness": sum(closeness_vals[1:])
                                / (model.config.n_instances - 1),
                                "A_matrix": wandb.Image(fig),
                            },
                            step=step,
                        )

            recon_loss = recon_loss.mean()
            sparsity_loss = sparsity_loss.mean()
            param_match_loss = param_match_loss.mean()

            if pretrained_model_path:
                loss = param_match_loss + sparsity_coeff * sparsity_loss
            else:
                loss = recon_loss + sparsity_coeff * sparsity_loss

        loss.backward()
        assert model.A.grad is not None
        # Don't update the gradient of the 0th instance (which we fixed to be the identity)
        model.A.grad[0] = torch.zeros_like(model.A.grad[0])
        opt.step()

    torch.save(model.state_dict(), out_dir / "model.pth")
    if config.wandb_project:
        wandb.save(str(out_dir / "model.pth"))


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = (
            f"sp{config.max_sparsity_coeff}_"
            f"lr{config.lr}_"
            f"p{config.pnorm}_"
            f"bs{config.batch_size}_"
            f"ft{config.n_features}_"
            f"hid{config.n_hidden}"
        )
    return config.wandb_run_name_prefix + run_suffix


def main(
    config_path_or_obj: Path | str | Config, sweep_config_path: Path | str | None = None
) -> None:
    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, sweep_config_path)
        # Save the config to wandb
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "final_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config.model_dump(mode="json"), f, indent=2)
            wandb.save(str(config_path), policy="now", base_path=tmp_dir)
            # Unfortunately wandb.save is async, so we need to wait for it to finish before
            # continuing, and wandb python api provides no way to do this.
            # TODO: Find a better way to do this.
            time.sleep(1)

    set_seed(config.seed)
    logger.info(config)

    run_name = get_run_name(config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(config=config, device=device)

    optimize(
        model=model,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model_path=config.pretrained_model_path,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
