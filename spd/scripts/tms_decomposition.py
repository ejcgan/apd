# %%
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc
from matplotlib import colors as mcolors
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange

from spd.utils import calculate_closeness_to_identity, permute_to_identity


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
    max_sparsity_coeff: float
    sparsity_loss_type: Literal["jacobian", "dotted"] = "jacobian"
    sparsity_warmup_pct: float = 0.0
    bias_val: float = 0.0
    train_bias: bool = False


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_probability: torch.Tensor | None = None,
        importance: torch.Tensor | None = None,
        device: str = "cuda",
        bias_val: float = 0.0,
        train_bias: bool = False,
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

        bias_data = (
            torch.zeros((config.n_instances, config.n_features), device=device, requires_grad=False)
            + bias_val
        )
        self.b_final = nn.Parameter(bias_data) if train_bias else bias_data

        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the output and intermediate hidden states."""
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        # A: [instance, n_features, k]
        # B: [instance, k, n_hidden]

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


def plot_intro_diagram(
    weight: torch.Tensor,
    filepath: Path | None = None,
    pos_quadrant_only: bool = False,
    closeness_vals: list[str] | None = None,
) -> None:
    sel = range(config.n_instances)  # can be used to highlight specific sparsity levels

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

        if pos_quadrant_only:
            ax.set_xlim((0, z))
            ax.set_ylim((0, z))
            ax.spines["left"].set_position(("data", 0))
            ax.spines["bottom"].set_position(("data", 0))
        else:
            ax.set_xlim((-z, z))
            ax.set_ylim((-z, z))
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_position("center")

        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Write the closeness_val at the very top of the plot
        if closeness_vals is not None:
            ax.text(0.5, 1.1, closeness_vals[i], ha="center", va="center")

    plt.show()
    if filepath is not None:
        plt.savefig(filepath)


def optimize(
    model: Model,
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 1e-3,
    lr_scale: Callable[[int, int], float] = linear_lr,
    pnorm: float = 0.75,
    max_sparsity_coeff: float = 0.02,
    sparsity_loss_type: Literal["jacobian", "dotted"] = "jacobian",
    sparsity_warmup_pct: float = 0.0,
) -> None:
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    def get_sparsity_coeff_linear_warmup(step: int) -> float:
        warmup_steps = int(steps * sparsity_warmup_pct)
        if step < warmup_steps:
            return max_sparsity_coeff * (step / warmup_steps)
        return max_sparsity_coeff

    with trange(steps) as t:
        for step in t:
            step_lr = lr * lr_scale(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out, h_0, h_1, hidden, pre_relu = model(batch)

            # Reconstruction loss
            error = model.importance * (batch.abs() - out) ** 2
            recon_loss = einops.reduce(error, "b i f -> i", "mean")

            if sparsity_loss_type == "dotted":
                out_dotted = model.importance * torch.einsum("bih,bih->bi", batch, out).sum()
                grad_hidden, grad_pre_relu = torch.autograd.grad(
                    out_dotted, (hidden, pre_relu), create_graph=True
                )
                grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)
                sparsity_loss = (
                    (grad_h_0 * h_0) ** 2 + 1e-16
                    # (grad_h_0 * h_0 + grad_h_1 * h_1) ** 2 + 1e-16
                ).sqrt()
            elif sparsity_loss_type == "jacobian":
                # The above sparsity loss calculates the gradient on a single output direction. We
                # want the gradient on all output dimensions
                sparsity_loss = torch.zeros_like(h_0, requires_grad=True)
                for feature_idx in range(out.shape[-1]):
                    grad_hidden, grad_pre_relu = torch.autograd.grad(
                        out[:, :, feature_idx].sum(),
                        (hidden, pre_relu),
                        grad_outputs=torch.tensor(1.0, device=out.device),
                        retain_graph=True,
                    )
                    grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                    grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)

                    # sparsity_inner = grad_h_0 * h_0 + grad_h_1 * h_1
                    sparsity_inner = grad_h_0 * h_0

                    sparsity_loss = sparsity_loss + sparsity_inner**2
                sparsity_loss = (sparsity_loss / out.shape[-1] + 1e-16).sqrt()
            else:
                raise ValueError(f"Unknown sparsity loss type: {sparsity_loss_type}")

            sparsity_loss = einops.reduce(
                (sparsity_loss.abs() ** pnorm).sum(dim=-1), "b i -> i", "mean"
            )

            if step % print_freq == print_freq - 1 or step == 0:
                sparsity_repr = [f"{x:.4f}" for x in sparsity_loss]
                recon_repr = [f"{x:.4f}" for x in recon_loss]
                tqdm.write(f"Sparsity loss: \n{sparsity_repr}")
                tqdm.write(f"Reconstruction loss: \n{recon_repr}")
            recon_loss = recon_loss.sum()
            sparsity_loss = sparsity_loss.sum()

            sparsity_coeff = get_sparsity_coeff_linear_warmup(step)
            loss = recon_loss + sparsity_coeff * sparsity_loss

            loss.backward()
            opt.step()
            # Force the A matrix to have norm 1 in the second last dimension (the hidden dimension)
            model.A.data = model.A.data / model.A.data.norm(p=2, dim=-2, keepdim=True)

            if step % print_freq == print_freq - 1 or step == 0:
                closeness_vals: list[str] = []
                for i in range(model.config.n_instances):
                    permuted_matrix = permute_to_identity(model.A[i])
                    closeness = calculate_closeness_to_identity(permuted_matrix)
                    closeness_vals.append(f"{closeness:.4f}")

                tqdm.write(f"W after {step + 1} steps (before gradient update)")
                plot_intro_diagram(weight=model.A.detach() @ model.B.detach())
                tqdm.write(f"B after {step + 1} steps (before gradient update)")
                plot_intro_diagram(
                    weight=model.B.detach(),
                    closeness_vals=closeness_vals,
                )
                tqdm.write(f"W after {step + 1} steps (before gradient update) abs")
                prev_n_instances = model.config.n_instances
                model.config.n_instances = 8
                plot_intro_diagram(
                    weight=torch.abs(model.A.detach() @ model.B.detach())[:8],
                    pos_quadrant_only=True,
                )
                tqdm.write(f"B after {step + 1} steps (before gradient update) abs")
                plot_intro_diagram(
                    weight=torch.abs(model.B.detach())[:8],
                    pos_quadrant_only=True,
                    closeness_vals=closeness_vals,
                )
                model.config.n_instances = prev_n_instances


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # %%

    # Set torch seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    config = Config(
        n_features=5,
        n_hidden=2,
        n_instances=15,
        k=5,
        n_batch=1024,
        steps=40_000,
        print_freq=5000,
        lr=1e-3,
        lr_scale=cosine_decay_lr,
        pnorm=0.75,
        max_sparsity_coeff=0.02,
        sparsity_loss_type="jacobian",
        sparsity_warmup_pct=0.0,
        bias_val=0.0,
        train_bias=False,
    )

    model = Model(
        config=config,
        device=device,
        feature_probability=torch.tensor([1 / 20])[:],
        bias_val=config.bias_val,
        train_bias=config.train_bias,
    )
    print("Plot of B at initialization")
    plot_intro_diagram(weight=model.B.detach())

    optimize(
        model,
        n_batch=config.n_batch,
        steps=config.steps,
        print_freq=config.print_freq,
        lr=config.lr,
        lr_scale=config.lr_scale,
        pnorm=config.pnorm,
        max_sparsity_coeff=config.max_sparsity_coeff,
        sparsity_loss_type=config.sparsity_loss_type,
        sparsity_warmup_pct=config.sparsity_warmup_pct,
    )

# %%
