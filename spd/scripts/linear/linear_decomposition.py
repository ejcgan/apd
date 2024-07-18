"""Linear decomposition script."""

import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import einops
import fire
import numpy as np
import torch
import wandb
import yaml
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from spd.log import logger
from spd.models import DeepLinearComponentModel, DeepLinearModel
from spd.types import RootPath
from spd.utils import (
    init_wandb,
    load_config,
    set_seed,
)

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_run_name_prefix: str = ""
    seed: int = 0
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


def optimize(
    dlc_model: DeepLinearComponentModel,
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

    pretrained_weights: list[torch.Tensor] | None = None
    if pretrained_model_path:
        pretrained_model = DeepLinearModel.from_pretrained(pretrained_model_path).to(device)
        pretrained_weights = [weight for weight in pretrained_model.layers]
        assert len(pretrained_weights) == dlc_model.n_layers
        for weight in pretrained_weights:
            weight.requires_grad = False

    opt = torch.optim.Adam(dlc_model.parameters(), lr=config.lr)

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

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            dlc_out, layer_acts, inner_acts = dlc_model(batch)
            assert len(layer_acts) == dlc_model.n_layers
            assert len(inner_acts) == dlc_model.n_layers

            param_match_loss = torch.zeros(dlc_model.n_instances, device=device)
            if pretrained_model_path:
                assert pretrained_weights is not None
                # If the user passed a pretrained model, then calculate the param_match_loss
                # Get the Frobenius norm between the pretrained weight and the current model's W
                for i in range(dlc_model.n_layers):
                    AB = torch.einsum("ifk,ikg->ifg", dlc_model.layers[i].A, dlc_model.layers[i].B)
                    param_match_loss = (
                        param_match_loss
                        + ((AB - pretrained_weights[i]) ** 2).sum(dim=(-2, -1)).sqrt()
                    )

            output_error = (dlc_out - batch) ** 2
            out_recon_loss = einops.reduce(output_error, "b i f -> i", "mean")

            all_Bs = [dlc_model.layers[i].B for i in range(dlc_model.n_layers)]

            sparsity_loss = torch.zeros_like(layer_acts[0], requires_grad=True)
            for feature_idx in range(dlc_out.shape[-1]):
                grad_layer_acts = torch.autograd.grad(
                    dlc_out[:, :, feature_idx].sum(),
                    layer_acts,
                    grad_outputs=torch.tensor(1.0, device=dlc_out.device),
                    retain_graph=True,
                    allow_unused=True,
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
            sparsity_loss = (sparsity_loss / dlc_out.shape[-1] + 1e-16).sqrt()

            sparsity_loss = einops.reduce(
                ((sparsity_loss.abs() + 1e-16) ** current_pnorm).sum(dim=-1), "b i -> i", "mean"
            )

            # sparsity_loss = torch.tensor([0.0 for _ in range(dlc_model.n_instances)], device=device)
            if step % config.print_freq == config.print_freq - 1 or step == 0:
                # sparsity_repr = [f"{x:.4f}" for x in sparsity_loss]
                # recon_repr = [f"{x:.4f}" for x in recon_loss]
                sparsity_repr = [f"{x}" for x in sparsity_loss]
                recon_repr = [f"{x}" for x in out_recon_loss]
                tqdm.write(f"Step {step}")
                tqdm.write(f"Current pnorm: {current_pnorm}")
                tqdm.write(f"Sparsity loss: \n{sparsity_repr}")
                tqdm.write(f"Reconstruction loss: \n{recon_repr}")
                if pretrained_model_path:
                    # param_match_repr = [f"{x:.4f}" for x in param_match_loss]
                    param_match_repr = [f"{x}" for x in param_match_loss]
                    tqdm.write(f"Param match loss: \n{param_match_repr}")

                if config.wandb_project:
                    wandb.log(
                        {
                            "step": step,
                            "current_pnorm": current_pnorm,
                            "current_lr": step_lr,
                            "sparsity_loss": sparsity_loss.mean().item(),
                            "recon_loss": out_recon_loss.mean().item(),
                            "param_match_loss": param_match_loss.mean().item(),
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
        ), "n_features, n_layers, and n_instances must be set"
    dlc_model = DeepLinearComponentModel(
        n_features=n_features, n_layers=n_layers, n_instances=n_instances, k=config.k
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
