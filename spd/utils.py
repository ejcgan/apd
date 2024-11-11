import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import einops
import numpy as np
import torch
import wandb
import yaml
from dotenv import load_dotenv
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from wandb.apis.public import Run

from spd.models.base import Model, SPDFullRankModel, SPDModel, SPDRankPenaltyModel
from spd.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)
Q = TypeVar("Q")


def to_root_path(path: str | Path):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


def from_root_path(path: str | Path) -> Path:
    """Converts absolute paths to relative ones, relative to the repo root."""
    path = Path(path)
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        # If the path is not relative to REPO_ROOT, return the original path
        return path


def permute_to_identity(x: torch.Tensor, normalize_rows: bool = False) -> torch.Tensor:
    """Permute the rows of a matrix such that the maximum value in each column is on the leading
    diagonal.

    Args:
        x: The input matrix.
        normalize_rows: Whether to normalize the rows of the output matrix.
    """

    # Assert that arr only has two dimensions and that it is square
    assert x.dim() == 2
    assert x.shape[0] == x.shape[1], "Must have the same number of subnetworks (k) as features"

    # Get the number of rows and columns
    n_rows, n_cols = x.shape

    # Find the row index of the maximum value in each column
    max_row_indices_raw = torch.argmax(x, dim=0).tolist()

    # Get the indices of the non unique max_row_indices
    unique_indices = set()
    duplicate_indices = []
    for i in range(n_rows):
        if max_row_indices_raw[i] in unique_indices:
            duplicate_indices.append(i)
        else:
            unique_indices.add(max_row_indices_raw[i])

    remaining_indices = [i for i in range(n_rows) if i not in unique_indices]
    # Now we want to swap out the duplicate indices with any remaining indices
    for i in range(len(duplicate_indices)):
        max_row_indices_raw[duplicate_indices[i]] = remaining_indices[i]

    # Ensure that we output a permuted version and have no duplicate rows
    assert set(max_row_indices_raw) == set(range(n_rows))

    out_rows = x[max_row_indices_raw]

    if normalize_rows:
        out_rows = out_rows / out_rows.norm(dim=1, p=2, keepdim=True)
    return out_rows


def calculate_closeness_to_identity(x: Float[Tensor, "... a b"]) -> float:
    """Frobenius norm of the difference between the input matrix and the identity matrix.

    If x has more than two dimensions, the result is meaned over all but the final two dimensions.
    """
    eye = torch.eye(n=x.shape[-2], m=x.shape[-1], device=x.device)
    return torch.norm(x - eye, p="fro", dim=(-2, -1)).mean().item()


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def load_config(config_path_or_obj: Path | str | T, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(config_path_or_obj).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)
    return config_model(**config_dict)


BaseModelType = TypeVar("BaseModelType", bound=BaseModel)


def replace_pydantic_model(model: BaseModelType, *updates: dict[str, Any]) -> BaseModelType:
    """Create a new model with (potentially nested) updates in the form of dictionaries.

    Args:
        model: The model to update.
        updates: The zero or more dictionaries of updates that will be applied sequentially.

    Returns:
        A replica of the model with the updates applied.

    Examples:
        >>> class Foo(BaseModel):
        ...     a: int
        ...     b: int
        >>> foo = Foo(a=1, b=2)
        >>> foo2 = replace_pydantic_model(foo, {"a": 3})
        >>> foo2
        Foo(a=3, b=2)
        >>> class Bar(BaseModel):
        ...     foo: Foo
        >>> bar = Bar(foo={"a": 1, "b": 2})
        >>> bar2 = replace_pydantic_model(bar, {"foo": {"a": 3}})
        >>> bar2
        Bar(foo=Foo(a=3, b=2))
    """
    return model.__class__(**deep_update(model.model_dump(), *updates))


def init_wandb(config: T, project: str, sweep_config_path: Path | str | None) -> T:
    """Initialize Weights & Biases and return a config updated with sweep hyperparameters.

    If no sweep config is provided, the config is returned as is.

    If a sweep config is provided, wandb is first initialized with the sweep config. This will
    cause wandb to choose specific hyperparameters for this instance of the sweep and store them
    in wandb.config. We then update the config with these hyperparameters.

    Args:
        config: The base config.
        project: The name of the wandb project.
        sweep_config_path: The path to the sweep config file. If provided, updates the config with
            the hyperparameters from this instance of the sweep.

    Returns:
        Config updated with sweep hyperparameters (if any).
    """
    if sweep_config_path is not None:
        with open(sweep_config_path) as f:
            sweep_data = yaml.safe_load(f)
        wandb.init(config=sweep_data, save_code=True)
    else:
        load_dotenv(override=True)
        wandb.init(project=project, entity=os.getenv("WANDB_ENTITY"), save_code=True)

    # Update the config with the hyperparameters for this sweep (if any)
    config = replace_pydantic_model(config, wandb.config)

    # Update the non-frozen keys in the wandb config (only relevant for sweeps)
    wandb.config.update(config.model_dump(mode="json"))
    return config


def save_config_to_wandb(config: BaseModel, filename: str = "final_config.yaml") -> None:
    # Save the config to wandb
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / filename
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, indent=2)
        wandb.save(str(config_path), policy="now", base_path=tmp_dir)
        # Unfortunately wandb.save is async, so we need to wait for it to finish before
        # continuing, and wandb python api provides no way to do this.
        # TODO: Find a better way to do this.
        time.sleep(1)


def init_param_(param: torch.Tensor, scale: float = 1.0) -> None:
    torch.nn.init.kaiming_uniform_(param)
    with torch.no_grad():
        param.mul_(scale)


class DatasetGeneratedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[Q],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(  # type: ignore
        self,
    ) -> Iterator[Q]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # type: ignore


class BatchedDataLoader(DataLoader[Q], Generic[Q]):
    """DataLoader that unpacks the batch in __getitem__.

    This is used for datasets which generate a whole batch in one call to __getitem__.
    """

    def __init__(
        self,
        dataset: Dataset[Q],
        num_workers: int = 0,
    ):
        super().__init__(dataset, num_workers=num_workers)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # type: ignore
        for batch, label in super().__iter__():
            yield batch[0], label[0]


def calc_grad_attributions_rank_one(
    out: Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"],
    inner_acts_vals: list[Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]],
) -> Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the element-wise product of the gradient of the output dimension w.r.t. the
    inner acts and the inner acts themselves.

    Note: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes. Unrelatedly, we use retain_graph=True in a bunch of cases
    where we want to later use the `out` variable in e.g. the loss function.

    Args:
        out: The output of the model.
        inner_acts_vals: The inner acts of the model (i.e. the set of subnetwork activations for
            each parameter matrix).

    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    attribution_scores: Float[Tensor, " k"] | Float[Tensor, "n_instances k"] = torch.zeros_like(
        inner_acts_vals[0]
    )
    for feature_idx in range(out.shape[-1]):
        feature_attributions: Float[Tensor, " k"] | Float[Tensor, "n_instances k"] = (
            torch.zeros_like(inner_acts_vals[0])
        )
        feature_grads: tuple[
            Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"], ...
        ] = torch.autograd.grad(out[..., feature_idx].sum(), inner_acts_vals, retain_graph=True)
        assert len(feature_grads) == len(inner_acts_vals)
        for param_matrix_idx in range(len(inner_acts_vals)):
            feature_attributions += (
                feature_grads[param_matrix_idx] * inner_acts_vals[param_matrix_idx]
            )

        attribution_scores += feature_attributions**2

    return attribution_scores


def calc_grad_attributions_rank_one_per_layer(
    out: Float[Tensor, "... d_out"],
    inner_acts_vals: list[Float[Tensor, "... k"]],
) -> list[Float[Tensor, "... k"]]:
    """Calculate the gradient attributions for each layer.

    This differs from calc_attributions_rank_one in that it returns a list of attribution scores
    for each layer that are the sum of squares of output features, rather than taking the sum of
    squared attributions across layers.

    i.e. we do sum_{n_features}(activation * grad)^2 for each layer, rather than
    sum_{n_features}(sum_{layers}(activation * grad))^2.

    Note that we don't have an ablation version of this for computational reasons.
    Args:
        out: The output of the model.
        inner_acts: The inner acts of the model (i.e. the set of subnetwork activations for each
            parameter matrix).

    Returns:
        The list of attribution scores for each layer.
    """

    layer_attribution_scores: list[Float[Tensor, "... k"]] = [
        torch.zeros_like(inner_acts_vals[0]) for _ in range(len(inner_acts_vals))
    ]
    for feature_idx in range(out.shape[-1]):
        feature_grads: tuple[Float[Tensor, "... k"], ...] = torch.autograd.grad(
            out[..., feature_idx].sum(), inner_acts_vals, retain_graph=True
        )
        assert len(feature_grads) == len(inner_acts_vals)
        for param_matrix_idx in range(len(inner_acts_vals)):
            layer_attribution_scores[param_matrix_idx] += (
                feature_grads[param_matrix_idx] * inner_acts_vals[param_matrix_idx]
            ).pow(2)

    return layer_attribution_scores


def calc_grad_attributions_full_rank(
    out: Float[Tensor, "... out_dim"],
    inner_acts: dict[str, Float[Tensor, "... k d_out"]],
    layer_acts: dict[str, Float[Tensor, "... d_out"]],
) -> Float[Tensor, "... k"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the element-wise product of the gradient of the output dimension w.r.t. the
    layer acts and the inner acts.

    Note: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes. Unrelatedly, we use retain_graph=True in a bunch of cases
    where we want to later use the `out` variable in e.g. the loss function.

    Args:
        out: The output of the model.
        inner_acts: The activations at the output of each subnetwork before being summed.
        layer_acts: The activations at the output of each layer after being summed.

    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    assert inner_acts.keys() == layer_acts.keys()
    first_param_matrix_name = next(iter(inner_acts.keys()))
    attribution_scores: Float[Tensor, "... k"] = torch.zeros(
        inner_acts[first_param_matrix_name].shape[:-1],
        device=inner_acts[first_param_matrix_name].device,
    )
    out_dim = out.shape[-1]
    for feature_idx in range(out_dim):
        feature_attributions: Float[Tensor, "... k"] = torch.zeros(
            inner_acts[first_param_matrix_name].shape[:-1],
            device=inner_acts[first_param_matrix_name].device,
        )
        grad_layer_acts: tuple[Float[Tensor, "... d_out"], ...] = torch.autograd.grad(
            out[..., feature_idx].sum(), list(layer_acts.values()), retain_graph=True
        )
        for i, param_matrix_name in enumerate(layer_acts.keys()):
            # Note that this operation would be equivalent to:
            # einsum(grad_inner_acts, inner_acts, "... k d_out ,... k d_out -> ... k")
            # since the gradient distributes over the sum.
            feature_attributions += einops.einsum(
                grad_layer_acts[i].detach(),
                inner_acts[param_matrix_name],
                "... d_out ,... k d_out -> ... k",
            )

        attribution_scores += feature_attributions**2

    return attribution_scores


def calc_grad_attributions_full_rank_per_layer(
    out: Float[Tensor, "... out_dim"],
    inner_acts: dict[str, Float[Tensor, "... k d_out"]],
    layer_acts: dict[str, Float[Tensor, "... d_out"]],
) -> list[Float[Tensor, "... k"]]:
    """Calculate the attributions for each layer.

    This differs from calc_attributions_full_rank in that it returns a list of attribution scores
    for each layer that are the sum of squares of output features, rather than taking the sum of
    squared attributions across layers.

    i.e. we do sum_{n_features}(activation * grad)^2 for each layer, rather than
    sum_{n_features}(sum_{layers}(activation * grad))^2.

    Note that we don't have an ablation version of this for computational reasons.

    Args:
        out: The output of the model.
        inner_acts: The activations at the output of each subnetwork before being summed.
        layer_acts: The activations at the output of each layer after being summed.

    Returns:
        The list of attribution scores for each layer.
    """
    first_param_matrix_name = next(iter(inner_acts.keys()))
    layer_attribution_scores: list[Float[Tensor, "... k"]] = [
        torch.zeros(
            inner_acts[first_param_matrix_name].shape[:-1],
            device=inner_acts[first_param_matrix_name].device,
        )
        for _ in range(len(inner_acts))
    ]
    for feature_idx in range(out.shape[-1]):
        grad_layer_acts: tuple[Float[Tensor, "... k"], ...] = torch.autograd.grad(
            out[..., feature_idx].sum(), list(layer_acts.values()), retain_graph=True
        )
        for i, param_matrix_name in enumerate(layer_acts.keys()):
            layer_attribution_scores[i] += einops.einsum(
                grad_layer_acts[i].detach(),
                inner_acts[param_matrix_name],
                "... d_out ,... k d_out -> ... k",
            ).pow(2)

    return layer_attribution_scores


@torch.inference_mode()
def calc_ablation_attributions(
    model: SPDModel | SPDFullRankModel | SPDRankPenaltyModel,
    batch: Float[Tensor, "batch ... n_features"],
    out: Float[Tensor, "batch ... d_model_out"],
) -> Float[Tensor, "batch ... k"]:
    """Calculate the attributions by ablating each subnetwork one at a time."""

    attr_shape = out.shape[:-1] + (model.k,)  # (batch, k) or (batch, n_instances, k)
    attributions = torch.zeros(attr_shape, device=out.device, dtype=out.dtype)
    for subnet_idx in range(model.k):
        stored_vals = model.set_subnet_to_zero(subnet_idx)
        ablation_out, _, _ = model(batch)
        out_recon = ((out - ablation_out) ** 2).mean(dim=-1)
        attributions[..., subnet_idx] = out_recon
        model.restore_subnet(subnet_idx, stored_vals)
    return attributions


def calc_activation_attributions(
    inner_acts: dict[
        str, Float[Tensor, "batch k d_out"] | Float[Tensor, "batch n_instances k d_out"]
    ],
) -> Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]:
    """Calculate the attributions by taking the L2 norm of the activations in each subnetwork.

    Args:
        inner_acts: The activations at the output of each subnetwork before being summed.
    Returns:
        The attributions for each subnetwork.
    """
    first_param = inner_acts[next(iter(inner_acts.keys()))]
    assert len(first_param.shape) in (3, 4)

    attribution_scores: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"] = (
        torch.zeros(first_param.shape[:-1], device=first_param.device, dtype=first_param.dtype)
    )
    for param_matrix in inner_acts.values():
        attribution_scores += param_matrix.pow(2).sum(dim=-1)
    return attribution_scores


def calculate_attributions(
    model: SPDModel | SPDFullRankModel | SPDRankPenaltyModel,
    batch: Float[Tensor, "... n_features"],
    out: Float[Tensor, "... n_features"],
    inner_acts: dict[str, Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"]],
    layer_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
    attribution_type: Literal["ablation", "gradient", "activation"],
    spd_type: Literal["rank_one", "full_rank", "rank_penalty"],
) -> Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"]:
    attributions = None
    if attribution_type == "ablation":
        attributions = calc_ablation_attributions(model=model, batch=batch, out=out)
    elif attribution_type == "gradient":
        if spd_type == "rank_one":
            attributions = calc_grad_attributions_rank_one(
                out=out, inner_acts_vals=list(inner_acts.values())
            )
        else:
            attributions = calc_grad_attributions_full_rank(
                out=out, inner_acts=inner_acts, layer_acts=layer_acts
            )
    elif attribution_type == "activation":
        assert spd_type != "rank_one", "Activation attributions not supported for rank one"
        attributions = calc_activation_attributions(inner_acts=inner_acts)
    else:
        raise ValueError(f"Invalid attribution type: {attribution_type}")
    return attributions


def calc_topk_mask(
    attribution_scores: Float[Tensor, "batch ... k"],
    topk: float,
    batch_topk: bool,
) -> Float[Tensor, "batch ... k"]:
    """Calculate the top-k mask.

    Args:
        attribution_scores: The attribution scores to calculate the top-k mask for.
        topk: The number of top-k elements to select. If `batch_topk` is True, this is multiplied
            by the batch size to get the number of top-k elements over the whole batch.
        batch_topk: If True, the top-k mask is calculated over the concatenated batch and k
            dimensions.

    Returns:
        The top-k mask.
    """
    batch_size = attribution_scores.shape[0]
    topk = int(topk * batch_size) if batch_topk else int(topk)

    if batch_topk:
        attribution_scores = einops.rearrange(attribution_scores, "b ... k -> ... (b k)")

    topk_indices = attribution_scores.topk(topk, dim=-1).indices
    topk_mask = torch.zeros_like(attribution_scores, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=topk_indices, value=True)

    if batch_topk:
        topk_mask = einops.rearrange(topk_mask, "... (b k) -> b ... k", b=batch_size)

    return topk_mask


def find_key_for_value(dictionary: dict[Any, Any], target_value: Any):
    for key, value_list in dictionary.items():
        if target_value in value_list:
            return key
    return None  # Return None if the value is not found in any list


def calc_neuron_indices(
    neuron_permutations: tuple[Int[Tensor, "..."], ...],
    num_neurons: int,
    num_functions: int,
):
    """Create a list of length n_layers, where each element is a list of length num_functions.
    The ith element of this list is a list of the indices of the neurons in the corresponding
    layer of the controlled piecewise linear that connect to the ith function."""
    all_neurons = torch.concatenate(neuron_permutations)
    assert torch.all(all_neurons.sort().values == torch.arange(len(all_neurons)))

    n_layers = len(neuron_permutations)
    ordered_neurons_dict = {
        i: torch.arange(i * num_neurons, (i + 1) * num_neurons) for i in range(num_functions)
    }
    neuron_indices = [
        [
            torch.tensor(
                [
                    index
                    for index, neuron in enumerate(neuron_permutations[layer].numpy())
                    if find_key_for_value(ordered_neurons_dict, neuron) == function
                ],
                dtype=torch.int64,
            )
            for function in range(num_functions)
        ]
        for layer in range(n_layers)
    ]
    return neuron_indices


@torch.inference_mode()
def remove_grad_parallel_to_subnetwork_vecs(
    A: Float[Tensor, "... d_in _"], A_grad: Float[Tensor, "... d_in _"]
) -> None:
    """Modify the gradient by subtracting it's component parallel to the activation.

    I.e. subtract the projection of the gradient vector onto the activation vector.

    This is to stop Adam from changing the norm of A. Note that this will not completely prevent
    Adam from changing the norm due to Adam's (m/(sqrt(v) + eps)) term not preserving the norm
    direction.

    The final dimension is k in the case of rank one and m in the case of full rank.
    """
    parallel_component = einops.einsum(A_grad, A, "... d_in _, ... d_in _ -> ... _")
    A_grad -= einops.einsum(parallel_component, A, "... _, ... d_in _ -> ... d_in _")


class SPDOutputs(NamedTuple):
    target_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"] | None
    )
    spd_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
    )
    spd_topk_model_output: (
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
    )
    layer_acts: dict[str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]]
    inner_acts: dict[
        str,
        Float[Tensor, "batch k d_out"]  # full rank
        | Float[Tensor, "batch n_instances k d_out"]  # full rank
        | Float[Tensor, "batch k"]  # rank one
        | Float[Tensor, "batch n_instances k"],  # rank one
    ]
    attribution_scores: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]
    topk_mask: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]


def run_spd_forward_pass(
    spd_model: SPDModel | SPDFullRankModel | SPDRankPenaltyModel,
    target_model: Model | None,
    input_array: Float[Tensor, "batch n_inputs"],
    attribution_type: Literal["gradient", "ablation", "activation"],
    spd_type: Literal["rank_one", "full_rank", "rank_penalty"],
    batch_topk: bool,
    topk: float,
    distil_from_target: bool,
) -> SPDOutputs:
    # non-SPD model, and SPD-model non-topk forward pass
    if target_model is not None:
        target_model_output, _, _ = target_model(input_array)
    else:
        target_model_output = None

    model_output_spd, layer_acts, inner_acts = spd_model(input_array)
    attribution_scores = calculate_attributions(
        model=spd_model,
        batch=input_array,
        out=model_output_spd,
        inner_acts=inner_acts,
        layer_acts=layer_acts,
        attribution_type=attribution_type,
        spd_type=spd_type,
    )

    # We always assume the final subnetwork is the one we want to distil
    topk_attrs = attribution_scores[..., :-1] if distil_from_target else attribution_scores
    topk_mask = calc_topk_mask(topk_attrs, topk, batch_topk=batch_topk)
    if distil_from_target:
        # Add back the final subnetwork index to the topk mask and set it to True
        last_subnet_mask = torch.ones(
            (*topk_mask.shape[:-1], 1), dtype=torch.bool, device=attribution_scores.device
        )
        topk_mask = torch.cat((topk_mask, last_subnet_mask), dim=-1)

    model_output_spd_topk, _, _ = spd_model(input_array, topk_mask=topk_mask)
    attribution_scores = attribution_scores.cpu().detach()
    return SPDOutputs(
        target_model_output=target_model_output,
        spd_model_output=model_output_spd,
        spd_topk_model_output=model_output_spd_topk,
        layer_acts=layer_acts,
        inner_acts=inner_acts,
        attribution_scores=attribution_scores,
        topk_mask=topk_mask,
    )


def download_wandb_file(run: Run, file_name: str) -> Path:
    cache_dir = Path(os.environ.get("SPD_CACHE_DIR", "/tmp/"))
    run_cache_dir = cache_dir / run.id
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    file_on_wandb = run.file(file_name)
    return Path(file_on_wandb.download(exist_ok=True, replace=True, root=run_cache_dir).name)  # type: ignore


def load_yaml(file_path: Path) -> dict[str, Any]:
    with open(file_path) as f:
        return yaml.safe_load(f)
