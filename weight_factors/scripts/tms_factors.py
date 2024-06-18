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
from tqdm import tqdm, trange


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
    # Number of parameter factors
    k: int
    n_batch: int
    steps: int
    print_freq: int
    lr: float
    lr_scale: Callable[[int, int], float]
    pnorm: float
    sparsity_coeff: float


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
        # self.W = nn.Parameter(
        #     torch.empty((config.n_instances, config.n_features, config.n_hidden), device=device)
        # )
        self.A = nn.Parameter(
            torch.empty((config.n_instances, config.n_features, config.k), device=device)
        )
        self.B = nn.Parameter(
            torch.empty((config.n_instances, config.k, config.n_hidden), device=device)
        )
        # nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)
        self.b_final = nn.Parameter(
            torch.zeros((config.n_instances, config.n_features), device=device)
        )

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the output and intermediate hidden states."""
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        # A: [instance, n_features, k]
        # B: [instance, k, n_hidden]

        h_0 = torch.einsum("...if,ifk->...ik", features, self.A)
        hidden = torch.einsum("...ik,ikh->...ih", h_0, self.B)

        h_1 = torch.einsum("...ih,ikh->...ik", hidden, self.B)
        out = torch.einsum("...ik,ifk->...if", h_1, self.A)

        # hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        # out = torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out, h_0, h_1

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


def optimize(
    model: Model,
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 1e-3,
    lr_scale: Callable[[int, int], float] = linear_lr,
    pnorm: float = 0.5,
    sparsity_coeff: float = 0.001,
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
            out, h_0, h_1 = model(batch)

            # Reconstruction loss
            error = model.importance * (batch.abs() - out) ** 2
            recon_loss = einops.reduce(error, "b i f -> i", "mean").sum()

            # Sparsity loss
            out_dotted = (model.importance * (out**2)).sum()
            # Get the gradient of out_dotted w.r.t h_0 and h_1
            grad_h_0, grad_h_1 = torch.autograd.grad(out_dotted, (h_0, h_1), create_graph=True)
            sparsity_inner = grad_h_0.detach() * h_0 + grad_h_1.detach() * h_1  # batch, instance, k

            sparsity_loss = einops.reduce(
                sparsity_inner.norm(p=pnorm, dim=-1), "b i -> i", "mean"
            ).sum()

            loss = recon_loss + sparsity_coeff * sparsity_loss
            if step % print_freq == 0:
                tqdm.write(f"Reconstruction loss: {recon_loss.item()}")
                tqdm.write(f"Sparsity loss: {sparsity_loss.item()}")
                # tqdm.write(f"sparsity_inner final instance: {sparsity_inner[:5, -1, :]}")
                # tqdm.write(f"grad_h_0 times h_0: {grad_h_0[:5, -1, :] * h_0[:5, -1, :]}")
                # tqdm.write(f"h_0 final instance: {h_0[:5, -1, :]}")
            # loss = einops.reduce(error, "b i f -> i", "mean").sum()
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


def plot_intro_diagram(model: Model, weight: torch.Tensor, filepath: Path) -> None:
    cfg = model.config
    N = len(weight[:, 0])
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels
    # plt.rcParams["axes.prop_cycle"] = plt.cycler(
    #     "color",
    #     plt.cm.viridis(model.importance[0].cpu().numpy()),  # type: ignore
    # )
    plt.rcParams["figure.dpi"] = 200
    fig, axs = plt.subplots(1, len(sel), figsize=(2 * len(sel), 2))
    for i, ax in zip(sel, axs, strict=False):
        W = weight[i].cpu().detach().numpy()
        colors = [mcolors.to_rgba(c) for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
        # ax.scatter(W[:, 0], W[:, 1], c=colors[0 : len(W[:, 0])])
        ax.scatter(W[:, 0], W[:, 1])
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
    # plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set torch seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # %%
    config = Config(
        n_features=5,
        n_hidden=2,
        n_instances=10,
        k=5,
        n_batch=1024,
        steps=20_000,
        print_freq=100,
        lr=1e-3,
        lr_scale=cosine_decay_lr,
        pnorm=0.5,
        sparsity_coeff=0.01,
    )

    model = Model(
        config=config,
        device=device,
        # Exponential feature importance curve from 1 to 1/100
        importance=(0.9 ** torch.arange(config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        feature_probability=(20 ** -torch.linspace(0, 1, config.n_instances))[:, None],
    )
    optimize(
        model,
        n_batch=config.n_batch,
        steps=config.steps,
        print_freq=config.print_freq,
        lr=config.lr,
        lr_scale=config.lr_scale,
        pnorm=config.pnorm,
        sparsity_coeff=config.sparsity_coeff,
    )
    # %%
    print("Plot of W")
    plot_intro_diagram(
        model,
        weight=model.A.detach() @ model.B.detach(),
        # filepath=Path(__file__).parent / "out" / "tms_factors_features_W.png",
        filepath=Path(__file__).parent
        / "out"
        / f"tms_factors_features_W_pnorm-{config.pnorm}_sparsity-{config.sparsity_coeff}_samples-{config.n_instances}.png",
    )
    print("Plot of B")
    plot_intro_diagram(
        model,
        weight=model.B.detach(),
        filepath=Path(__file__).parent
        / "out"
        / f"tms_factors_features_B_pnorm-{config.pnorm}_sparsity-{config.sparsity_coeff}_samples-{config.n_instances}.png",
    )


# %%
