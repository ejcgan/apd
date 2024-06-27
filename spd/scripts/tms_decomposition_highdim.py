# %%
from collections.abc import Callable
from dataclasses import dataclass

import einops
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
    bias_val: float | None = None
    train_bias: bool = False
    sparsity_warmup_pct: float = 0.0


class Model(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_probability: torch.Tensor | None = None,
        importance: torch.Tensor | None = None,
        device: str = "cuda",
        bias_val: float | None = None,
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

        bias_vals = torch.zeros(
            (config.n_instances, config.n_features), device=device, requires_grad=False
        )
        self.b_final = nn.Parameter(bias_vals) if train_bias else bias_vals

        # nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)

        if bias_val is not None:
            print(f"Setting bias to a constant value of {bias_val}")
            self.b_final.data = torch.ones_like(self.b_final) * bias_val

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

            # The above sparsity loss calculates the gradient on a single output direction. We want
            # the gradient on all output dimensions
            assert pnorm == 1, "Currently only pnorm=1 is supported for sparsity loss."
            sparsity_loss = torch.zeros(*h_0.shape, device=out.device, requires_grad=True)
            for feature_idx in range(out.shape[-1]):
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
                    (grad_h_0 * h_0) ** 2 + 1e-16
                    #  (grad_h_1 * h_1) ** 2 + 1e-16
                    # (grad_h_0 * h_0) + (grad_h_1 * h_1) + 1e-16
                ).sqrt()
                sparsity_loss = sparsity_loss + sparsity_inner
                # sparsity_loss = sparsity_loss + sparsity_inner**2
            # sparsity_loss = (sparsity_loss / out.shape[-1] + 1e-16).sqrt()
            # sparsity_loss = sparsity_loss / out.shape[-1] + 1e-16

            # Just h_0 as the sparsity penalty
            # sparsity_loss = h_1.abs()
            # sparsity_loss = h_0.abs()

            sparsity_loss = einops.reduce(
                (sparsity_loss.abs() ** pnorm).sum(dim=-1), "b i -> i", "mean"
            )
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
                tqdm.write(f"Reconstruction loss: {recon_loss.item()}")
                tqdm.write(f"Sparsity loss: {sparsity_loss.item()}")

                closeness_vals: list[str] = []
                for i in range(model.config.n_instances):
                    permuted_matrix = permute_to_identity(model.A[i], normalize_rows=True)
                    closeness = calculate_closeness_to_identity(permuted_matrix)
                    closeness_vals.append(f"{closeness:.4f}")
                tqdm.write(f"Closeness to identity: {closeness_vals}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # %%

    # Set torch seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    config = Config(
        n_features=20,
        n_hidden=10,
        n_instances=8,
        k=5,
        n_batch=1024,
        steps=40_000,
        print_freq=1000,
        lr=1e-3,
        lr_scale=cosine_decay_lr,
        pnorm=1,
        max_sparsity_coeff=0.02,
        # sparsity_warmup_pct=0.2,
        # bias_val=-0.1,
        train_bias=True,
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
        bias_val=config.bias_val,
        train_bias=config.train_bias,
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


# %%
