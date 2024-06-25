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
    max_sparsity_coeff: float
    init_file: str | None = None
    bias_file: str | None = None
    bias_val: float | None = None
    sparsity_warmup_pct: float = 0.0


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_probability: torch.Tensor | None = None,
        importance: torch.Tensor | None = None,
        device: str = "cuda",
        init_file: str | None = None,
        bias_file: str | None = None,
        bias_val: float | None = None,
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

        self.b_final = torch.zeros(
            (config.n_instances, config.n_features), device=device, requires_grad=False
        )
        # nn.init.xavier_normal_(self.W)
        if init_file is None:
            nn.init.xavier_normal_(self.A)
            nn.init.xavier_normal_(self.B)
        else:
            print("Initializing A to the identity matrix and B to the W from the initial run")
            # Init A to the the identity in n_features and k, repeaet over the n_instance dimension
            weight_info = torch.load(init_file)
            self.A.data = (
                torch.eye(config.n_features, config.k).repeat(config.n_instances, 1, 1).to(device)
            )
            # # # init B to be the same as W from the initial run
            self.B.data = weight_info["A"] @ weight_info["B"]

        if bias_file is not None:
            print(f"Loading bias from {bias_file}")
            loaded_bias = torch.load(bias_file)
            assert loaded_bias.shape == self.b_final.shape
            assert loaded_bias.abs().sum() > 0
            self.b_final.data = loaded_bias
        elif bias_val is not None:
            print(f"Setting bias to a constant value of {bias_val}")
            self.b_final = torch.ones_like(self.b_final) * bias_val

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

        # hidden = torch.einsum("...if,ifh->...ih", features, self.W)
        # out = torch.einsum("...ih,ifh->...if", hidden, self.W)
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
    model: Model,
    weight: torch.Tensor,
    filepath: Path | None = None,
    pos_quadrant_only: bool = False,
) -> None:
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
    sparsity_warmup_pct: float = 0.0,
) -> None:
    hooks = []
    cfg = model.config

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

            # # Sparsity loss (in the direction of the output)
            # out_dotted = (model.importance * (out**2)).sum() / 2
            # # Sparsity loss (in the direction of the true labels)
            # out_dotted = model.importance * torch.einsum("b i h,b i h->b i", batch, out).sum()

            # # # Get the gradient of out_dotted w.r.t h_0 and h_1
            # grad_h_0, grad_h_1 = torch.autograd.grad(out_dotted, (h_0, h_1), create_graph=True)
            # grad_hidden, grad_pre_relu = torch.autograd.grad(
            #     out_dotted, (hidden, pre_relu), create_graph=True
            # )
            # grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
            # grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)
            # sparsity_loss = (
            #     # ((grad_h_0.detach() * h_0) + (grad_h_1.detach() * h_1) + 1e-16).abs()
            #     (grad_h_0 * h_0) ** 2 + 1e-16
            #     # (h_0) ** 2 + 1e-16
            #     #   (grad_h_1) ** 2 + 1e-16
            #     #   (grad_h_1 * h_1) ** 2 + 1e-16
            #     #  (grad_h_0 * h_0) ** 2 + 1e-16
            #     #   (grad_h_1.detach()) ** 2 + 1e-16
            #     #   (grad_h_1) ** 2 + 1e-16
            # ).sqrt()

            # sparsity_loss = h_0.abs()

            # The above sparsity loss calculates the gradient on a single output direction. We want the gradient on all
            # output dimensions
            sparsity_loss = torch.zeros(out.shape[1], device=out.device, requires_grad=True)

            # sparsity_loss = 0
            for feature_idx in range(out.shape[-1]):
                # grad_h_0, grad_h_1 = torch.autograd.grad(
                #     out[:, :, feature_idx].sum(),
                #     (h_0, h_1),
                #     grad_outputs=torch.tensor(1.0, device=out.device),
                #     retain_graph=True,
                # )
                grad_hidden, grad_pre_relu = torch.autograd.grad(
                    out[:, :, feature_idx].sum(),
                    (hidden, pre_relu),
                    grad_outputs=torch.tensor(1.0, device=out.device),
                    retain_graph=True,
                )
                grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)
                sparsity_inner = (
                    # (grad_h_0.detach() * h_0) ** 2 + (grad_h_1.detach() * h_1) ** 2 + 1e-16
                    # (grad_h_1.detach()) ** 2 + 1e-16
                    # (grad_h_0.detach() * h_0) ** 2 + 1e-16
                    # (grad_h_1.detach() * h_1) ** 2 + 1e-16
                    ###
                    # (grad_h_0 * h_0) ** 2 + 1e-16
                    #  (grad_h_1 * h_1) ** 2 + 1e-16
                    (grad_h_0 * h_0) + (grad_h_1 * h_1) + 1e-16
                )
                sparsity_inner = einops.reduce(
                    (sparsity_inner.abs() ** pnorm).sum(dim=-1), "b i -> i", "mean"
                )
                sparsity_loss = sparsity_loss + sparsity_inner
                # sparsity_loss = sparsity_loss + sparsity_inner**2
            # sparsity_loss = (sparsity_loss / out.shape[-1] + 1e-16).sqrt()
            # sparsity_loss = sparsity_loss / out.shape[-1] + 1e-16

            # Just h_0 as the sparsity penalty
            # sparsity_loss = h_1.abs()
            # sparsity_loss = h_0.abs()

            # sparsity_loss = einops.reduce(
            #     (sparsity_loss.abs() ** pnorm).sum(dim=-1), "b i -> i", "mean"
            # )
            # Do the above pnorm but take to the power of pnorm
            if step % print_freq == print_freq - 1 or step == 0:
                sparsity_repr = [f"{x:.4f}" for x in sparsity_loss]
                recon_repr = [f"{x:.4f}" for x in recon_loss]
                tqdm.write(f"Sparsity loss: \n{sparsity_repr}")
                tqdm.write(f"Reconstruction loss: \n{recon_repr}")
            recon_loss = recon_loss.sum()
            sparsity_loss = sparsity_loss.sum()

            sparsity_coeff = get_sparsity_coeff_linear_warmup(step)
            loss = recon_loss + sparsity_coeff * sparsity_loss

            # tqdm.write(f"sparsity_inner final instance: {sparsity_inner[:5, -1, :]}")
            # tqdm.write(f"grad_h_0 times h_0: {grad_h_0[:5, -1, :] * h_0[:5, -1, :]}")
            # tqdm.write(f"h_0 final instance: {h_0[:5, -1, :]}")
            # loss = einops.reduce(error, "b i f -> i", "mean").sum()
            loss.backward()
            opt.step()
            # Force the A matrix to have norm 1 in the second last dimension (the hidden dimension)
            model.A.data = model.A.data / model.A.data.norm(p=2, dim=-2, keepdim=True)
            # model.B.data = model.B.data / model.B.data.norm(p=2, dim=-1, keepdim=True)
            if step % print_freq == print_freq - 1 or step == 0:
                # tqdm.write(f"Reconstruction loss: {recon_loss.item()}")
                # tqdm.write(f"Sparsity loss: {sparsity_loss.item()}")
                tqdm.write(f"W after {step + 1} steps (before gradient update)")
                plot_intro_diagram(
                    model,
                    weight=model.A.detach() @ model.B.detach(),
                )
                tqdm.write(f"B after {step + 1} steps (before gradient update)")
                plot_intro_diagram(
                    model,
                    weight=model.B.detach(),
                )
                tqdm.write(f"W after {step + 1} steps (before gradient update) abs")
                prev_n_instances = model.config.n_instances
                model.config.n_instances = 8
                plot_intro_diagram(
                    model,
                    # weight=model.A.detach() @ model.B.detach(),
                    weight=torch.abs(model.A.detach() @ model.B.detach())[:8],
                    pos_quadrant_only=True,
                )
                tqdm.write(f"B after {step + 1} steps (before gradient update) abs")
                plot_intro_diagram(
                    model,
                    # weight=model.B.detach(),
                    weight=torch.abs(model.B.detach())[:8],
                    pos_quadrant_only=True,
                )
                model.config.n_instances = prev_n_instances


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # %%

    # Set torch seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    config = Config(
        n_features=20,
        n_hidden=10,
        n_instances=15,
        k=5,
        n_batch=1024,
        steps=40_000,
        print_freq=5000,
        lr=5e-2,
        lr_scale=cosine_decay_lr,
        pnorm=0.75,
        max_sparsity_coeff=0.002,
        # sparsity_warmup_pct=0.2,
        # init_file="tms_factors_features.pt",
        # bias_file="b_final.pt",
        # bias_val=-0.1,
    )

    model = Model(
        config=config,
        device=device,
        # Exponential feature importance curve from 1 to 1/100
        # importance=(0.9 ** torch.arange(config.n_features))[None, :],
        # importance=(0.9 ** torch.arange(config.n_features))[None, :],
        # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
        # feature_probability=(20 ** -torch.linspace(0, 1, config.n_instances))[:, None],
        # feature_probability=torch.tensor([1 / 20])[:, None],
        feature_probability=torch.tensor([1 / 20])[:],
        # init_file=config.init_file,
        # bias_file=config.bias_file,
        bias_val=config.bias_val,
    )
    # print("Plot of initial W")
    # plot_intro_diagram(
    #     model,
    #     weight=torch.load("tms_factors_features_W.pt"),
    # )
    print("Plot of B at initialization")
    plot_intro_diagram(
        model,
        weight=model.B.detach(),
    )

    optimize(
        model,
        n_batch=config.n_batch,
        steps=config.steps,
        print_freq=config.print_freq,
        lr=config.lr,
        lr_scale=config.lr_scale,
        pnorm=config.pnorm,
        max_sparsity_coeff=config.max_sparsity_coeff,
        sparsity_warmup_pct=config.sparsity_warmup_pct,
    )
    # Store the weight matrix and bias
    # weight_info = {"A": model.A.detach(), "B": model.B.detach()}
    # weight_info = {"A": model.A.detach(), "B": model.B.detach(), "b_final": model.b_final.detach()}
    # torch.save(weight_info, "tms_factors_features.pt")

    # %%
    model.config.n_instances = 6
    print("Plot of W after training")
    plot_intro_diagram(
        model,
        # weight=(model.A.detach() @ model.B.detach())[6:9],
        # weight=torch.abs(model.A.detach() @ model.B.detach())[9:12],
        weight=torch.abs(model.A.detach() @ model.B.detach())[4:10],
    )
    print("Plot of B after training")
    plot_intro_diagram(
        model,
        # weight=torch.abs(model.B.detach())[9:12],
        weight=torch.abs(model.B.detach())[4:10],
    )


# %%
