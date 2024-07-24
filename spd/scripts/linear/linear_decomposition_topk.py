"""Linear decomposition script."""

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
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from tqdm import tqdm

from spd.log import logger
from spd.models import DeepLinearComponentModel, DeepLinearModel
from spd.types import RootPath
from spd.utils import (
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
    topk: int = 1
    batch_size: int
    steps: int
    print_freq: int
    save_freq: int | None = None
    lr: float
    # max_sparsity_coeff: float
    n_features: int | None = None
    n_layers: int | None = None
    n_instances: int | None = None
    k: int | None = None
    lr_scale: Literal["linear", "constant", "cosine"] = "constant"
    lr_warmup_pct: float = 0.0
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
    model: DeepLinearComponentModel, topk: int, device: str
) -> tuple[
    Float[Tensor, "batch n_instances n_features"], list[Float[Tensor, "batch n_instances k"]]
]:
    """
    Collect inner activation data for visualization.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the inner activations. It then permutes the activations to align with the identity.

    Args:
        model (DeepLinearComponentModel): The model to collect data from.
        topk (int): The number of topk values to collect.
        device (str): The device to run computations on.

    Returns:
        - The input test batch (identity matrix expanded over instance dimension).
        - A list of permuted inner activations for each layer.

    """
    test_batch = einops.repeat(
        torch.eye(model.n_features, device=device),
        "b f -> b i f",
        i=model.n_instances,
    )
    _, _, test_inner_acts_topk = model.forward_topk(test_batch, topk=topk)

    test_inner_acts_topk_permuted = []
    for layer in range(model.n_layers):
        test_inner_acts_topk_layer_permuted = []
        for i in range(model.n_instances):
            test_inner_acts_topk_layer_permuted.append(
                permute_to_identity(test_inner_acts_topk[layer][:, i, :].abs())
            )
        test_inner_acts_topk_permuted.append(
            torch.stack(test_inner_acts_topk_layer_permuted, dim=1)
        )

    return test_batch, test_inner_acts_topk_permuted


def optimize(
    dlc_model: DeepLinearComponentModel,
    config: Config,
    out_dir: Path,
    device: str,
    pretrained_model_path: RootPath | None = None,
) -> None:
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

        for group in opt.param_groups:
            group["lr"] = step_lr
        opt.zero_grad(set_to_none=True)
        batch = dlc_model.generate_batch(config.batch_size).to(dtype=torch.float32, device=device)

        total_samples += batch.shape[0]  # don't include the number of instances

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # Stage 1: Do a full forward pass and get the gradients w.r.t inner_acts
            dlc_out, _, inner_acts = dlc_model(batch)
            all_grads = [torch.zeros_like(inner_acts[i]) for i in range(dlc_model.n_layers)]
            for feature_idx in range(dlc_out.shape[-1]):
                grads = torch.autograd.grad(
                    dlc_out[:, :, feature_idx].sum(), inner_acts, retain_graph=True
                )
                for layer_idx in range(dlc_model.n_layers):
                    all_grads[layer_idx] += grads[layer_idx]

            # Stage 2: Do a forward pass with topk
            dlc_out_topk, _, inner_acts_topk = dlc_model.forward_topk(batch, config.topk, all_grads)
            assert len(inner_acts_topk) == dlc_model.n_layers

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

            output_topk_error = (dlc_out_topk - batch) ** 2
            out_recon_loss_topk = einops.reduce(output_topk_error, "b i f -> i", "mean")

            with torch.inference_mode():
                if step % config.print_freq == config.print_freq - 1 or step == 0:
                    recon_repr = [f"{x}" for x in out_recon_loss]
                    tqdm.write(f"Step {step}")
                    tqdm.write(f"Reconstruction loss: \n{recon_repr}")
                    if pretrained_model_path:
                        # param_match_repr = [f"{x:.4f}" for x in param_match_loss]
                        param_match_repr = [f"{x}" for x in param_match_loss]
                        tqdm.write(f"Param match loss: \n{param_match_repr}")

                    test_batch, test_inner_acts_topk = collect_inner_act_data(
                        dlc_model, config.topk, device
                    )

                    fig = plot_inner_acts(batch=test_batch, inner_acts=test_inner_acts_topk)
                    fig.savefig(out_dir / f"inner_acts_{step}.png")
                    plt.close(fig)
                    tqdm.write(f"Saved inner_acts to {out_dir / f'inner_acts_{step}.png'}")

                    if config.wandb_project:
                        wandb.log(
                            {
                                "step": step,
                                "current_lr": step_lr,
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
            out_recon_loss_topk = out_recon_loss_topk.mean()
            param_match_loss = param_match_loss.mean()

            # TODO: Add a coefficient between the two terms in these losses
            if pretrained_model_path:
                loss = param_match_loss + out_recon_loss_topk
            else:
                loss = out_recon_loss + out_recon_loss_topk

        loss.backward()
        opt.step()

    torch.save(dlc_model.state_dict(), out_dir / f"model_{config.steps}.pth")
    logger.info(f"Saved model to {out_dir / f'model_{config.steps}.pth'}")
    if config.wandb_project:
        wandb.save(str(out_dir / f"model_{config.steps}.pth"))


def get_run_name(config: Config) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = f"lr{config.lr}_bs{config.batch_size}_"
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
    print(f"Using device: {device}")
    if config.pretrained_model_path:
        dl_model = DeepLinearModel.from_pretrained(config.pretrained_model_path).to(device)
        assert (
            config.n_features is None and config.n_layers is None and config.n_instances is None
        ), "n_features, n_layers, and n_instances must not be set if pretrained_model_path is set"
        n_features = dl_model.n_features
        n_layers = dl_model.n_layers
        n_instances = dl_model.n_instances
    else:
        n_features, n_layers, n_instances = config.n_features, config.n_layers, config.n_instances
        assert (
            n_features is not None and n_layers is not None and n_instances is not None
        ), "n_features, n_layers, and n_instances must be set in the config if no pretrained model"
    dlc_model = DeepLinearComponentModel(
        n_features=n_features,
        n_layers=n_layers,
        n_instances=n_instances,
        k=config.k,
        topk=config.topk,
    ).to(device)

    optimize(
        dlc_model=dlc_model,
        config=config,
        out_dir=out_dir,
        device=device,
        pretrained_model_path=config.pretrained_model_path,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
