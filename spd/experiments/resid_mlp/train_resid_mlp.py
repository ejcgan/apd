"""Trains a residual linear model on one-hot input vectors."""

import json
from pathlib import Path
from typing import Literal, Self

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator
from torch import Tensor, nn
from tqdm import tqdm

from spd.experiments.resid_mlp.models import ResidualMLPModel
from spd.experiments.resid_mlp.resid_mlp_dataset import (
    ResidualMLPDataset,
)
from spd.run_spd import get_lr_schedule_fn
from spd.utils import DatasetGeneratedDataLoader, compute_feature_importances, set_seed

wandb.require("core")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: int = 0
    label_fn_seed: int = 0
    label_type: Literal["act_plus_resid", "abs"] = "act_plus_resid"
    use_trivial_label_coeffs: bool = False
    n_instances: PositiveInt
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"] = Field(
        description="Defines the activation function in the model. Also used in the labeling "
        "function if label_type is act_plus_resid."
    )
    apply_output_act_fn: bool
    in_bias: bool
    out_bias: bool
    feature_probability: PositiveFloat
    importance_val: float | None = None
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    fixed_random_embedding: bool = False
    fixed_identity_embedding: bool = False
    n_batches_final_losses: PositiveInt = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert not (
            self.fixed_random_embedding and self.fixed_identity_embedding
        ), "Can't have both fixed_random_embedding and fixed_identity_embedding"
        if self.fixed_identity_embedding:
            assert (
                self.n_features == self.d_embed
            ), "n_features must equal d_embed if we are using an identity embedding matrix"
        return self


def train(
    config: Config,
    model: ResidualMLPModel,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances d_resid"],
        ]
    ],
    feature_importances: Float[Tensor, "batch_size n_instances n_features"],
    device: str,
    out_dir: Path | None = None,
) -> Float[Tensor, " n_instances"]:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Add this line to get the lr_schedule_fn
    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule)

    current_losses = torch.tensor([])
    pbar = tqdm(range(config.steps), total=config.steps)
    for step, (batch, labels) in zip(pbar, dataloader, strict=False):
        if step >= config.steps:
            break

        # Add this block to update the learning rate
        current_lr = config.lr * lr_schedule_fn(step, config.steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch: Float[Tensor, "batch n_instances n_features"] = batch.to(device)
        labels: Float[Tensor, "batch n_instances n_features"] = labels.to(device)
        out, _, _ = model(batch)
        # Scale by feature importance as in Anthropic paper

        loss = ((out - labels) ** 2) * feature_importances
        loss = loss.mean(dim=(0, 2))
        current_losses = loss.detach()
        loss = loss.mean(dim=0)
        loss.backward()
        optimizer.step()
        if step % config.print_freq == 0:
            pbar.set_description(f"loss={current_losses.mean():.2e}, lr={current_lr:.2e}")

    if out_dir is not None:
        model_path = out_dir / "target_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        config_path = out_dir / "target_model_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        print(f"Saved config to {config_path}")

        # Save the coefficients used to generate the labels if label_type is act_plus_resid
        assert isinstance(dataloader.dataset, ResidualMLPDataset)
        assert dataloader.dataset.label_coeffs is not None
        label_coeffs = dataloader.dataset.label_coeffs.tolist()
        label_coeffs_path = out_dir / "label_coeffs.json"
        with open(label_coeffs_path, "w") as f:
            json.dump(label_coeffs, f)
        print(f"Saved label coefficients to {label_coeffs_path}")

    # Calculate final losses by averaging many batches
    final_losses = []
    for _ in range(config.n_batches_final_losses):
        batch, labels = next(iter(dataloader))
        batch = batch.to(device)
        labels = labels.to(device)
        out, _, _ = model(batch)
        loss = ((out - labels) ** 2) * feature_importances
        loss = loss.mean(dim=(0, 2))
        final_losses.append(loss)
    final_losses = torch.stack(final_losses).mean(dim=0).cpu().detach()
    print(f"Final losses: {final_losses.numpy()}")
    return final_losses


def run_train(config: Config, device: str) -> Float[Tensor, " n_instances"]:
    run_name = (
        f"resid_mlp_identity_{config.label_type}_n-instances{config.n_instances}_"
        f"n-features{config.n_features}_d-resid{config.d_embed}_"
        f"d-mlp{config.d_mlp}_n-layers{config.n_layers}_seed{config.seed}"
    )
    out_dir = Path(__file__).parent / "out" / run_name

    model = ResidualMLPModel(
        n_instances=config.n_instances,
        n_features=config.n_features,
        d_embed=config.d_embed,
        d_mlp=config.d_mlp,
        n_layers=config.n_layers,
        act_fn_name=config.act_fn_name,
        apply_output_act_fn=config.apply_output_act_fn,
        in_bias=config.in_bias,
        out_bias=config.out_bias,
    ).to(device)

    if config.fixed_random_embedding or config.fixed_identity_embedding:
        # Don't train the embedding matrices
        model.W_E.requires_grad = False
        model.W_U.requires_grad = False
        if config.fixed_random_embedding:
            # Init with randn values and make unit norm
            model.W_E.data[:, :, :] = torch.randn(
                config.n_instances, config.n_features, config.d_embed, device=device
            )
            model.W_E.data /= model.W_E.data.norm(dim=-1, keepdim=True)
            # Set W_U to W_E^T
            model.W_U.data = model.W_E.data.transpose(-2, -1)
        elif config.fixed_identity_embedding:
            assert config.n_features == config.d_embed, "n_features must equal d_embed for W_E=id"
            # Make W_E the identity matrix
            model.W_E.data[:, :, :] = einops.repeat(
                torch.eye(config.d_embed, device=device),
                "d_features d_embed -> n_instances d_features d_embed",
                n_instances=config.n_instances,
            )

    label_coeffs = None
    if config.use_trivial_label_coeffs:
        label_coeffs = torch.ones(config.n_instances, config.n_features, device=device)

    dataset = ResidualMLPDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
        calc_labels=True,
        label_type=config.label_type,
        act_fn_name=config.act_fn_name,
        label_fn_seed=config.label_fn_seed,
        label_coeffs=label_coeffs,
        data_generation_type=config.data_generation_type,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    feature_importances = compute_feature_importances(
        batch_size=config.batch_size,
        n_instances=config.n_instances,
        n_features=config.n_features,
        importance_val=config.importance_val,
        device=device,
    )

    final_losses = train(
        config=config,
        model=model,
        trainable_params=[p for p in model.parameters() if p.requires_grad],
        dataloader=dataloader,
        feature_importances=feature_importances,
        device=device,
        out_dir=out_dir,
    )
    return final_losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(
        seed=0,
        label_fn_seed=0,
        n_instances=10,
        n_features=100,
        d_embed=100,
        d_mlp=40,
        n_layers=1,
        act_fn_name="relu",
        apply_output_act_fn=True,
        label_type="abs",
        data_generation_type="at_least_zero_active",
        use_trivial_label_coeffs=True,
        in_bias=False,
        out_bias=False,
        feature_probability=0.5,
        importance_val=0.97,
        batch_size=256,
        steps=50_000,
        print_freq=100,
        lr=5e-3,
        lr_schedule="cosine",
    )

    set_seed(config.seed)

    run_train(config, device)
