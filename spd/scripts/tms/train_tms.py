# %%
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc
from matplotlib import colors as mcolors
from torch import nn
from torch.nn import functional as F
from tqdm import trange


# %%
@dataclass
class Config:
    n_features: int
    n_hidden: int

    # We optimize n_instances models in a single training loop
    # to let us sweep over sparsity or importance curves
    # efficiently.

    # We could potentially use torch.vmap instead.
    n_instances: int


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_probability: torch.Tensor | None = None,
        importance: torch.Tensor | None = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.config = config
        self.W = nn.Parameter(
            torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)
        )
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(
            torch.zeros((config.n_instances, config.n_features), device=device)
        )

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out

    def generate_batch(self, n_batch: int) -> torch.Tensor:
        feat = torch.rand(
            (n_batch, self.config.n_instances, self.config.n_features), device=self.W.device
        )
        batch = torch.where(
            torch.rand(
                (n_batch, self.config.n_instances, self.config.n_features), device=self.W.device
            )
            <= self.feature_probability,
            feat,
            torch.zeros((), device=self.W.device),
        )
        return batch


def linear_lr(step: int, steps: int) -> float:
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    return np.cos(0.5 * np.pi * step / (steps - 1))


def optimize(
    model: Model,
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 5e-3,
    lr_scale: Callable[[int, int], float] = linear_lr,
) -> None:
    hooks = []
    cfg = model.config

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    with trange(steps) as t:
        for step in t:
            step_lr = lr * lr_scale(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out = model(batch)
            error = model.importance * (batch.abs() - out) ** 2
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
                t.set_postfix(
                    loss=loss.item() / cfg.n_instances,
                    lr=step_lr,
                )


def plot_intro_diagram(model: Model, filepath: Path) -> None:
    cfg = model.config
    WA = model.W.detach()
    N = len(WA[:, 0])
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color",
        plt.cm.viridis(model.importance[0].cpu().numpy()),  # type: ignore
    )
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2))
    axs = np.array(axs)
    for i, ax in zip(sel, axs, strict=False):
        W = WA[i].cpu().detach().numpy()
        colors = [mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
        ax.scatter(W[:, 0], W[:, 1], c=colors[0 : len(W[:, 0])])
        ax.set_aspect("equal")
        ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W), W), axis=1), colors=colors))  # type: ignore

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # %%
    config = Config(
        n_features=5,
        n_hidden=2,
        n_instances=12,
    )

    model = Model(
        config=config,
        device=device,
        # Exponential feature importance curve from 1 to 1/100
        # importance=(0.9 ** torch.arange(config.n_features))[None, :],
        importance=(1.0 ** torch.arange(config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        # feature_probability=(20 ** -torch.linspace(0, 1, config.n_instances))[:, None],
        # Make all features appear with probability 1/20
        feature_probability=torch.ones((config.n_instances, config.n_features), device=device) / 20,
    )
    optimize(model)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"tms_n-features{config.n_features}_n-hidden{config.n_hidden}_"
        f"n-instances{config.n_instances}.pth"
    )
    torch.save(model.state_dict(), out_dir / run_name)
    print(f"Saved model to {out_dir / run_name}")
    # %%
    plot_intro_diagram(model, filepath=out_dir / run_name.replace(".pth", ".png"))
    print(f"Saved diagram to {out_dir / run_name.replace('.pth', '.png')}")

# %%
