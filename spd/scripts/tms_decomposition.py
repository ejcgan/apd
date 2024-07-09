# %%
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import einops
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import collections as mc
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, trange

from spd.utils import calculate_closeness_to_identity, permute_to_identity

# %%


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
    max_sparsity_coeff: float
    pnorm: float | None = None
    pnorm_end: float | None = None
    lr_scale: Callable[[int, int], float] | None = None
    lr_warmup_pct: float = 0.0
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
            torch.empty((config.n_instances, config.n_features, config.k), device=device),
        )
        # self.A = torch.zeros(
        #     (config.n_instances, config.n_features, config.k), device=device, requires_grad=False
        # )
        # Make A an identity for each n_instance
        # self.A = (
        #     torch.eye(config.n_features, device=device, requires_grad=False)
        #     .unsqueeze(0)
        #     .expand(config.n_instances, config.n_features, config.k)
        # )

        self.B = nn.Parameter(
            torch.empty((config.n_instances, config.k, config.n_hidden), device=device)
        )

        # Set A to an identity matrix
        # self.A = (
        #     torch.eye(config.n_features, device=device, requires_grad=False)
        #     .unsqueeze(0)
        #     .expand(config.n_instances, config.n_features, config.k)
        # )

        bias_data = torch.zeros((config.n_instances, config.n_features), device=device) + bias_val
        self.b_final = nn.Parameter(bias_data) if train_bias else bias_data

        nn.init.xavier_normal_(self.A)
        # Initialise one of the A instances to be an identity matrix
        self.A.data[0] = torch.eye(config.n_features, device=device)
        nn.init.xavier_normal_(self.B)

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the output and intermediate hidden states."""
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        # A: [instance, n_features, k]
        # B: [instance, k, n_hidden]

        normed_A = self.A / self.A.norm(p=2, dim=-2, keepdim=True)

        h_0 = torch.einsum("...if,ifk->...ik", features, normed_A)
        hidden = torch.einsum("...ik,ikh->...ih", h_0, self.B)

        h_1 = torch.einsum("...ih,ikh->...ik", hidden, self.B)
        hidden_2 = torch.einsum("...ik,ifk->...if", h_1, normed_A)

        pre_relu = hidden_2 + self.b_final
        out = F.relu(pre_relu)
        return out, h_0, h_1, hidden, pre_relu, normed_A

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


def plot_A_matrix(
    x: torch.Tensor,
    model: Model,
    step: int,
    out_dir: Path,
    layout: str = "row",
    pos_only: bool = False,
) -> None:
    normed_A = x / x.norm(p=2, dim=-2, keepdim=True)
    # A_abs = normed_A.abs()
    # A_abs[A_abs < 0.001] = 0
    # A_abs = normed_A

    n_instances = normed_A.shape[0]

    if layout == "column":
        fig, axs = plt.subplots(
            n_instances, 1, figsize=(3, 1.5 * n_instances), squeeze=False, sharex=True
        )
    elif layout == "row":
        fig, axs = plt.subplots(
            1, n_instances, figsize=(2.5 * n_instances, 2), squeeze=False, sharey=True
        )
    else:
        raise ValueError("Layout must be either 'column' or 'row'")

    max_abs_val = normed_A.abs().max()
    vmin = -max_abs_val if not pos_only else 0
    vmax = max_abs_val
    cmap = "Blues" if pos_only else "RdBu"
    im = None
    for i in range(n_instances):
        if layout == "column":
            ax = axs[i, 0]
            im = ax.matshow(
                normed_A[i, :, :].T.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
            if i == 0:
                ax.xaxis.set_label_position("top")
                ax.set_xlabel("n_features")
            if i == n_instances - 1:
                ax.xaxis.set_ticks_position("bottom")
        else:  # layout == 'row'
            ax = axs[0, i]
            im = ax.matshow(
                normed_A[i, :, :].T.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.xaxis.set_ticks_position("bottom")
            if i == 0:
                ax.set_ylabel("k", rotation=0, labelpad=10, va="center")
            else:
                ax.set_yticks([])  # Remove y-axis ticks for all but the first plot
            # Put xlabel on the top
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("n_features")

    assert im is not None

    if layout == "column":
        # plt.tight_layout()
        plt.subplots_adjust(
            hspace=0.1, left=0.2
        )  # Adjust left margin and reduce space between plots
    else:  # layout == 'row'
        # plt.tight_layout()
        plt.subplots_adjust(
            wspace=0.1, bottom=0.15, top=0.9
        )  # Reduce space between plots and adjust margins
        # Add space at the bottom for the colorbar
        fig.subplots_adjust(bottom=0.2)

        # Add a narrower colorbar at the bottom
        cbar_ax = fig.add_axes((0.3, 0.05, 0.4, 0.02))  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    plt.savefig(
        out_dir
        / f"A_{step}_n_feats-{model.config.n_features}_n_hid-{model.config.n_hidden}_{layout}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
    tqdm.write(
        f"Saved to {out_dir / f'A_{step}_n_feats-{model.config.n_features}_n_hid-{model.config.n_hidden}_{layout}.png'}"
    )


def optimize(
    model: Model,
    n_batch: int = 1024,
    steps: int = 10_000,
    print_freq: int = 100,
    lr: float = 1e-3,
    lr_scale: Callable[[int, int], float] | None = None,
    pnorm: float | None = None,
    pnorm_end: float | None = None,
    max_sparsity_coeff: float = 0.02,
    sparsity_loss_type: Literal["jacobian", "dotted"] = "jacobian",
    sparsity_warmup_pct: float = 0.0,
    lr_warmup_pct: float = 0.0,
) -> tuple[float, float, float]:
    assert (pnorm is None and pnorm_end is not None) or (
        pnorm is not None and pnorm_end is None
    ), "Exactly one of pnorm and pnorm_end must be set"
    assert pnorm_end is not None or pnorm is not None, "pnorm_end must be set if pnorm is not set"

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)
    # opt = torch.optim.SGD(list(model.parameters()), lr=lr)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_sparsity_coeff_linear_warmup(step: int) -> float:
        warmup_steps = int(steps * sparsity_warmup_pct)
        if step < warmup_steps:
            return max_sparsity_coeff * (step / warmup_steps)
        return max_sparsity_coeff

    def get_lr_with_warmup(step: int) -> float:
        warmup_steps = int(steps * lr_warmup_pct)
        if step < warmup_steps:
            return lr * (step / warmup_steps)
        return lr if lr_scale is None else lr * lr_scale(step - warmup_steps, steps - warmup_steps)

    final_sparsity_loss = 0.0
    final_recon_loss = 0.0
    final_closeness = 0.0

    with trange(steps) as t:
        for step in t:
            step_lr = get_lr_with_warmup(step)

            current_pnorm = get_current_pnorm(step, steps, pnorm_end) if pnorm is None else pnorm

            for group in opt.param_groups:
                group["lr"] = step_lr
            opt.zero_grad(set_to_none=True)
            batch = model.generate_batch(n_batch)
            out, h_0, h_1, hidden, pre_relu, normed_A = model(batch)

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
                    # (grad_h_0 * h_0) ** 2 + 1e-16
                    (h_0) ** 2 + 1e-16
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

                    sparsity_inner = grad_h_0 * h_0 + grad_h_1 * h_1

                    sparsity_loss = sparsity_loss + sparsity_inner**2
                sparsity_loss = (sparsity_loss / out.shape[-1] + 1e-16).sqrt()
            else:
                raise ValueError(f"Unknown sparsity loss type: {sparsity_loss_type}")

            sparsity_loss = einops.reduce(
                ((sparsity_loss.abs() + 1e-16) ** current_pnorm).sum(dim=-1), "b i -> i", "mean"
            )

            if step % print_freq == print_freq - 1 or step == 0:
                sparsity_repr = [f"{x:.4f}" for x in sparsity_loss]
                recon_repr = [f"{x:.4f}" for x in recon_loss]
                tqdm.write(f"Current pnorm: {current_pnorm}")
                tqdm.write(f"Sparsity loss: \n{sparsity_repr}")
                tqdm.write(f"Reconstruction loss: \n{recon_repr}")
                closeness_vals: list[str] = []
                for i in range(model.config.n_instances):
                    permuted_matrix = permute_to_identity(model.A[i].T.abs())
                    closeness = calculate_closeness_to_identity(permuted_matrix)
                    closeness_vals.append(f"{closeness:.4f}")

                tqdm.write(f"W after {step + 1} steps (before gradient update)")
                normed_A = model.A / model.A.norm(p=2, dim=-2, keepdim=True)
                if model.config.n_hidden == 2:
                    plot_intro_diagram(
                        weight=normed_A @ model.B.detach(),
                        filepath=out_dir
                        / f"W_{step}_n_feats-{model.config.n_features}_n_hid-{model.config.n_hidden}.png",
                    )
                    tqdm.write(
                        f"Saved to {out_dir / f'W_{step}_n_feats-{model.config.n_features}_n_hid-{model.config.n_hidden}.png'}"
                    )

                # Permute the normed_A matrix to look like an identity matrix
                permuted_A_T_list = []
                for instance_idx in range(model.config.n_instances):
                    permuted_A_T_i = permute_to_identity(normed_A[instance_idx].T.abs())
                    permuted_A_T_list.append(permuted_A_T_i)
                permuted_A_T = torch.stack(permuted_A_T_list, dim=0)

                plot_A_matrix(permuted_A_T, model, step, out_dir, pos_only=True)

            recon_loss = recon_loss.sum()
            sparsity_loss = sparsity_loss.sum()

            sparsity_coeff = get_sparsity_coeff_linear_warmup(step)
            loss = recon_loss + sparsity_coeff * sparsity_loss

            loss.backward()
            assert model.A.grad is not None
            # Don't update the gradient of the 0th dimension of A
            model.A.grad[0] = torch.zeros_like(model.A.grad[0])
            opt.step()
            # Force the A matrix to have norm 1 in the second last dimension (the hidden dimension)
            # model.A.data = model.A.data / model.A.data.norm(p=2, dim=-2, keepdim=True)

            if step == steps - 1:  # Last step
                final_sparsity_loss = sparsity_loss.item() / model.config.n_instances
                final_recon_loss = recon_loss.item() / model.config.n_instances
                final_closeness = (
                    sum(
                        calculate_closeness_to_identity(permute_to_identity(model.A[i].T))
                        for i in range(model.config.n_instances)
                    )
                    / model.config.n_instances
                )

    return final_sparsity_loss, final_recon_loss, final_closeness


def run_sweep(config: Config, sparsity_coeffs: list[float]) -> list[dict[str, Any]]:
    results = []

    for coeff in tqdm(sparsity_coeffs, desc="Sparsity Coefficient Sweep"):
        config.max_sparsity_coeff = coeff
        model = Model(
            config=config,
            device=device,
            feature_probability=torch.tensor([1 / 20])[:],
            bias_val=config.bias_val,
            train_bias=config.train_bias,
        )

        final_sparsity_loss, final_recon_loss, final_closeness = optimize(
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

        results.append(
            {
                "coeff": coeff,
                "sparsity_loss": final_sparsity_loss,
                "recon_loss": final_recon_loss,
                "closeness": final_closeness,
            }
        )

    return results


def plot_results(results: list[dict[str, Any]]) -> None:
    coeffs = [r["coeff"] for r in results]
    sparsity_losses = [r["sparsity_loss"] for r in results]
    recon_losses = [r["recon_loss"] for r in results]
    closenesses = [r["closeness"] for r in results]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(coeffs, sparsity_losses, marker="o")
    plt.xscale("log")
    plt.title("Sparsity Loss")
    plt.xlabel("Sparsity Coefficient")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(coeffs, recon_losses, marker="o")
    plt.xscale("log")
    plt.title("Reconstruction Loss")
    plt.xlabel("Sparsity Coefficient")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(coeffs, closenesses, marker="o")
    plt.xscale("log")
    plt.title("Closeness to Identity")
    plt.xlabel("Sparsity Coefficient")
    plt.ylabel("Closeness")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # %%

    # Set torch seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    config = Config(
        n_features=10,
        n_hidden=5,
        n_instances=8,
        k=10,
        n_batch=2048,
        steps=40_000,
        print_freq=4000,
        lr=1e-2,
        pnorm=0.5,
        # pnorm_end=0.1,
        max_sparsity_coeff=0.0005,
        # lr_scale=cosine_decay_lr,
        lr_scale=None,
        lr_warmup_pct=0.1,
        sparsity_loss_type="jacobian",
        sparsity_warmup_pct=0.0,
        bias_val=0.0,
        train_bias=True,
    )

    # sparsity_coeffs = [0.0, 0.005, 0.01, 0.05, 0.1, 1.0]
    # results = run_sweep(config, sparsity_coeffs)
    # plot_results(results, out_file="sparsity_sweep_h0.png")

    model = Model(
        config=config,
        device=device,
        feature_probability=torch.tensor([1 / 20])[:],
        bias_val=config.bias_val,
        train_bias=config.train_bias,
    )

    final_sparsity_loss, final_recon_loss, final_closeness = optimize(
        model,
        n_batch=config.n_batch,
        steps=config.steps,
        print_freq=config.print_freq,
        lr=config.lr,
        lr_scale=config.lr_scale,
        pnorm=config.pnorm,
        pnorm_end=config.pnorm_end,
        max_sparsity_coeff=config.max_sparsity_coeff,
        sparsity_loss_type=config.sparsity_loss_type,
        sparsity_warmup_pct=config.sparsity_warmup_pct,
        lr_warmup_pct=config.lr_warmup_pct,
    )
    print(f"{final_sparsity_loss=} {final_recon_loss=} {final_closeness=}")

# %%
#
