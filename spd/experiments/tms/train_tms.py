import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib import collections as mc
from pydantic import BaseModel, PositiveInt
from torch import Tensor
from tqdm import tqdm, trange

from spd.experiments.tms.models import TMSModel
from spd.utils import DatasetGeneratedDataLoader, SparseFeatureDataset, set_seed


class TMSTrainConfig(BaseModel):
    n_features: PositiveInt
    n_hidden: PositiveInt

    # We optimize n_instances models in a single training loop
    # to let us sweep over sparsity or importance curves
    # efficiently.

    # We could potentially use torch.vmap instead.
    n_instances: PositiveInt
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr: float
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: TMSModel,
    dataloader: DatasetGeneratedDataLoader[tuple[torch.Tensor, torch.Tensor]],
    importance: float = 1.0,
    steps: int = 5_000,
    print_freq: int = 100,
    lr: float = 5e-3,
    lr_schedule: Callable[[int, int], float] = linear_lr,
) -> None:
    hooks = []

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    data_iter = iter(dataloader)
    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch, labels = next(data_iter)
            out, _, _ = model(batch)
            error = importance * (labels.abs() - out) ** 2
            loss = einops.reduce(error, "b i f -> i", "mean").sum()
            loss.backward()
            opt.step()

            if hooks:
                hook_data = dict(
                    model=model, step=step, opt=opt, error=error, loss=loss, lr=step_lr
                )
                for h in hooks:
                    h(hook_data)
            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item() / model.n_instances}")
                t.set_postfix(
                    loss=loss.item() / model.n_instances,
                    lr=step_lr,
                )


def plot_intro_diagram(model: TMSModel, filepath: Path) -> None:
    """2D polygon plot of the TMS model.

    Adapted from
    https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb.
    """
    WA = model.W.detach()
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels
    color = plt.cm.viridis(np.array([0.0]))  # type: ignore
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2))
    axs = np.array(axs)
    for i, ax in zip(sel, axs, strict=False):
        W = WA[i].cpu().detach().numpy()
        ax.scatter(W[:, 0], W[:, 1], c=color)
        ax.set_aspect("equal")
        ax.add_collection(
            mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=[color])  # type: ignore
        )

        z = 1.5
        ax.set_facecolor("#FCFBF8")
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))
        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position("center")
    plt.savefig(filepath)


def calculate_feature_cosine_similarities(
    model: TMSModel,
) -> Float[Tensor, " n_instances"]:
    """Calculate cosine similarities between feature vectors.

    Returns:
        tuple of (mean, min, max) cosine similarities for each instance
    """
    rows = model.W.detach()
    rows /= rows.norm(dim=-1, keepdim=True)
    cosine_sims = einops.einsum(rows, rows, "i f1 h, i f2 h -> i f1 f2")
    # Remove self-similarities from consideration
    mask = ~torch.eye(rows.shape[1], device=rows.device, dtype=torch.bool)
    masked_sims = cosine_sims[:, mask].reshape(rows.shape[0], -1)

    max_sims = masked_sims.max(dim=-1).values
    theoretical_max = (1 / model.n_hidden) ** 0.5

    if (max_sims > theoretical_max).any():
        warnings.warn(
            f"Maximum cosine similarity ({max_sims.max().item():.3f}) exceeds theoretical maximum "
            f"of 1/√d_hidden = {theoretical_max:.3f}",
            RuntimeWarning,
            stacklevel=2,
        )

    return max_sims


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TMSTrainConfig(
        n_features=5,
        n_hidden=2,
        n_instances=12,
        feature_probability=0.05,
        batch_size=1024,
        steps=5_000,
        seed=0,
        lr=5e-3,
        data_generation_type="at_least_zero_active",
    )

    set_seed(config.seed)

    model = TMSModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        n_hidden=config.n_hidden,
        device=device,
    )

    dataset = SparseFeatureDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        data_generation_type=config.data_generation_type,
        value_range=(0.0, 1.0),
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size)
    train(model, dataloader=dataloader, steps=config.steps)

    run_name = (
        f"tms_n-features{config.n_features}_n-hidden{config.n_hidden}_"
        f"n-instances{config.n_instances}_seed{config.seed}.pth"
    )
    out_dir = Path(__file__).parent / "out" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pth")
    print(f"Saved model to {out_dir / 'model.pth'}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=4)
    print(f"Saved config to {out_dir / 'config.json'}")

    if config.n_hidden == 2:
        plot_intro_diagram(model, filepath=out_dir / run_name.replace(".pth", ".png"))
        print(f"Saved diagram to {out_dir / run_name.replace('.pth', '.png')}")

    maxs = calculate_feature_cosine_similarities(model)
    print(f"Cosine sims max: {maxs.tolist()}")
    print(f"1/sqrt(n_hidden): {1 / np.sqrt(config.n_hidden)}")
