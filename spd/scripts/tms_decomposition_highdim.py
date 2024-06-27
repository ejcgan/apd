# %%
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
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
            torch.empty((config.n_instances, config.n_features, config.k), device=device),
        )

        self.B = nn.Parameter(
            torch.empty((config.n_instances, config.k, config.n_hidden), device=device)
        )

        # Set A to an identity matrix
        # self.A.data = (
        #     torch.eye(config.n_features, device=device)
        #     .unsqueeze(0)
        #     .expand(config.n_instances, config.n_features, config.k)
        # )
        bias_data = torch.zeros((config.n_instances, config.n_features), device=device) + bias_val
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
) -> tuple[float, float, float]:
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

    def get_sparsity_coeff_linear_warmup(step: int) -> float:
        warmup_steps = int(steps * sparsity_warmup_pct)
        if step < warmup_steps:
            return max_sparsity_coeff * (step / warmup_steps)
        return max_sparsity_coeff

    final_sparsity_loss = 0.0
    final_recon_loss = 0.0
    final_closeness = 0.0

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
                    # (grad_h_0 * h_0) ** 2 + 1e-16
                    (h_0) ** 2 + 1e-16
                    # (grad_h_0 * h_0 + grad_h_1 * h_1) ** 2 + 1e-16
                ).sqrt()
            elif sparsity_loss_type == "jacobian":
                # The above sparsity loss calculates the gradient on a single output direction. We
                # want the gradient on all output dimensions
                sparsity_loss = torch.zeros_like(h_0, requires_grad=True)
                for feature_idx in range(out.shape[-1]):
                    # grad_hidden, grad_pre_relu = torch.autograd.grad(
                    #     out[:, :, feature_idx].sum(),
                    #     (hidden, pre_relu),
                    #     grad_outputs=torch.tensor(1.0, device=out.device),
                    #     retain_graph=True,
                    # )
                    # grad_h_0 = torch.einsum("...ih,ikh->...ik", grad_hidden.detach(), model.B)
                    # grad_h_1 = torch.einsum("...if,ifk->...ik", grad_pre_relu.detach(), model.A)

                    # sparsity_inner = grad_h_0 * h_0 + grad_h_1 * h_1
                    # sparsity_inner = grad_h_0 * h_0
                    sparsity_inner = h_0

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
                closeness_vals: list[str] = []
                for i in range(model.config.n_instances):
                    permuted_matrix = permute_to_identity(model.A[i].T)
                    closeness = calculate_closeness_to_identity(permuted_matrix)
                    closeness_vals.append(f"{closeness:.4f}")
                tqdm.write(f"Closeness: \n{closeness_vals}")
                tqdm.write("\n")
            recon_loss = recon_loss.sum()
            sparsity_loss = sparsity_loss.sum()

            sparsity_coeff = get_sparsity_coeff_linear_warmup(step)
            loss = recon_loss + sparsity_coeff * sparsity_loss

            loss.backward()
            opt.step()
            # Force the A matrix to have norm 1 in the second last dimension (the hidden dimension)
            model.A.data = model.A.data / model.A.data.norm(p=2, dim=-2, keepdim=True)

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


def run_sweep(config: Config, sparsity_coeffs: list[float]) -> dict[str, float]:
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


def plot_results(results: dict[str, float]):
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
    torch.manual_seed(1)
    np.random.seed(1)

    config = Config(
        n_features=20,
        n_hidden=10,
        n_instances=8,
        k=20,
        n_batch=1024,
        steps=10_000,
        print_freq=2000,
        lr=1e-3,
        lr_scale=cosine_decay_lr,
        pnorm=0.9,
        max_sparsity_coeff=100,
        sparsity_loss_type="jacobian",
        sparsity_warmup_pct=0.0,
        bias_val=0.0,
        train_bias=False,
    )

    # sparsity_coeffs = [0.0, 0.005, 0.01, 0.05, 0.1, 1.0]
    # results = run_sweep(config, sparsity_coeffs)
    # plot_results(results, out_file="sparsity_sweep.png")

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
    print(f"{final_sparsity_loss=} {final_recon_loss=} {final_closeness=}")

# %%
