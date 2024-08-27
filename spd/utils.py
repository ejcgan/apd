import math
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

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

from spd.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)


def to_root_path(path: str | Path):
    """Converts relative paths to absolute ones, assuming they are relative to the rib root."""
    return Path(path) if Path(path).is_absolute() else Path(REPO_ROOT / path)


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


def init_param_(param: torch.Tensor) -> None:
    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))


class BatchedDataLoader(DataLoader[tuple[torch.Tensor, torch.Tensor]]):
    """DataLoader that generates batches by calling the dataset's `generate_batch` method."""

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        # assert that dataset has a generate_batch method
        assert hasattr(dataset, "generate_batch")
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __iter__(  # type: ignore
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(len(self)):
            yield self.dataset.generate_batch(self.batch_size)  # type: ignore


def calc_attributions(
    out: Float[Tensor, "... out_dim"], inner_acts: list[Float[Tensor, "... k"]]
) -> Float[Tensor, "... k"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the element-wise product of the gradient of the output dimension w.r.t. the
    inner acts and the inner acts themselves.

    Args:
        out: The output of the model.
        inner_acts: The inner acts of the model (i.e. the set of subnetwork activations for each
            parameter matrix).

    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    attribution_scores: Float[Tensor, "... k"] = torch.zeros_like(inner_acts[0])
    for feature_idx in range(out.shape[-1]):
        feature_attributions: Float[Tensor, "... k"] = torch.zeros_like(inner_acts[0])
        feature_grads: tuple[Float[Tensor, "... k"], ...] = torch.autograd.grad(
            out[..., feature_idx].sum(), inner_acts, retain_graph=True
        )
        assert len(feature_grads) == len(inner_acts)
        for param_matrix_idx in range(len(inner_acts)):
            feature_attributions += feature_grads[param_matrix_idx] * inner_acts[param_matrix_idx]

        attribution_scores += feature_attributions**2

    return attribution_scores


def calc_topk_mask(
    attribution_scores: Float[Tensor, "batch ... k"], topk: float, batch_topk: bool
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
