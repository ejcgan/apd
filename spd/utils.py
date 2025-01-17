import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import einops
import numpy as np
import torch
import yaml
from jaxtyping import Float, Int
from pydantic import BaseModel
from pydantic.v1.utils import deep_update
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from spd.models.base import Model, SPDFullRankModel, SPDModel, SPDRankPenaltyModel
from spd.settings import REPO_ROOT

T = TypeVar("T", bound=BaseModel)
Q = TypeVar("Q")

# Avoid seaborn package installation (sns.color_palette("colorblind").as_hex())
COLOR_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def to_root_path(path: str | Path) -> Path:
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


def init_param_(
    param: torch.Tensor,
    scale: float = 1.0,
    init_type: Literal["kaiming_uniform", "xavier_normal"] = "kaiming_uniform",
) -> None:
    if init_type == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(param)
        with torch.no_grad():
            param.mul_(scale)
    elif init_type == "xavier_normal":
        torch.nn.init.xavier_normal_(param, gain=scale)


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


def calc_grad_attributions(
    target_out: Float[Tensor, "batch out_dim"] | Float[Tensor, "batch n_instances out_dim"],
    pre_acts: dict[str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]],
    post_acts: dict[str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]],
    subnet_params: dict[
        str, Float[Tensor, "k d_in d_out"] | Float[Tensor, "n_instances k d_in d_out"]
    ],
    k: int,
) -> Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]:
    """Calculate the sum of the (squared) attributions from each output dimension.

    An attribution is the product of the gradient of the target model output w.r.t. the post acts
    and the inner acts (i.e. the output of each subnetwork before being summed).

    Note that we don't use the inner_acts collected from the SPD model, because this includes the
    computational graph of the full model. We only want the subnetwork parameters of the current
    layer to be in the computational graph. To do this, we multiply a detached version of the
    pre_acts by the subnet parameters.

    NOTE: Multplying the pre_acts by the subnet parameters would be less efficient than multiplying
    the pre_acts by A and then B in the case where subnet_params is rank one or rank penalty. In
    the future, we can implement this more efficient version. For now, this simpler version is fine.

    Note: This code may be run in between the training forward pass, and the loss.backward() and
    opt.step() calls; it must not mess with the training. The reason the current implementation is
    fine to run anywhere is that we just use autograd rather than backward which does not
    populate the .grad attributes. Unrelatedly, we use retain_graph=True in a bunch of cases
    where we want to later use the `out` variable in e.g. the loss function.

    Args:
        target_out: The output of the target model.
        pre_acts: The activations at the output of each subnetwork before being summed.
        post_acts: The activations at the output of each layer after being summed.
        subnet_params: The subnet parameter matrix at each layer.
        k: The number of subnetwork parameters.
    Returns:
        The sum of the (squared) attributions from each output dimension.
    """
    assert post_acts.keys() == pre_acts.keys() == subnet_params.keys()
    attr_shape = target_out.shape[:-1] + (k,)  # (batch, k) or (batch, n_instances, k)
    attribution_scores: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"] = (
        torch.zeros(attr_shape, device=target_out.device, dtype=target_out.dtype)
    )

    out_dim = target_out.shape[-1]
    for feature_idx in range(out_dim):
        feature_attributions: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"] = (
            torch.zeros(attr_shape, device=target_out.device, dtype=target_out.dtype)
        )
        grad_post_acts: tuple[
            Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"], ...
        ] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_acts.values()), retain_graph=True
        )
        for i, param_matrix_name in enumerate(post_acts.keys()):
            # Note that this operation would be equivalent to:
            # einsum(grad_inner_acts, inner_acts, "... k d_out ,... k d_out -> ... k")
            # since the gradient distributes over the sum.
            inner_acts = einops.einsum(
                pre_acts[param_matrix_name].detach().clone(),
                subnet_params[param_matrix_name],
                "... d_in, ... k d_in d_out -> ... k d_out",
            )
            feature_attributions += einops.einsum(
                grad_post_acts[i], inner_acts, "... d_out ,... k d_out -> ... k"
            )

        attribution_scores += feature_attributions**2

    return attribution_scores


def calc_grad_attributions_full_rank_per_layer(
    target_out: Float[Tensor, "batch out_dim"] | Float[Tensor, "batch n_instances out_dim"],
    pre_acts: dict[str, Float[Tensor, "batch d_in"] | Float[Tensor, "batch n_instances d_in"]],
    post_acts: dict[str, Float[Tensor, "batch d_out"] | Float[Tensor, "batch n_instances d_out"]],
    subnet_params: dict[
        str, Float[Tensor, "k d_in d_out"] | Float[Tensor, "n_instances k d_in d_out"]
    ],
    k: int,
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
        pre_acts: The activations at the output of each subnetwork before being summed.
        subnet_params: The subnet parameter matrix at each layer.
        layer_acts: The activations at the output of each layer after being summed.
        k: The number of subnetwork parameters.

    Returns:
        The list of attribution scores for each layer.
    """
    assert post_acts.keys() == pre_acts.keys() == subnet_params.keys()
    attr_shape = target_out.shape[:-1] + (k,)  # (batch, k) or (batch, n_instances, k)
    layer_attribution_scores: list[Float[Tensor, "... k"]] = [
        torch.zeros(attr_shape, device=target_out.device) for _ in range(len(pre_acts))
    ]
    for feature_idx in range(target_out.shape[-1]):
        grad_post_acts: tuple[Float[Tensor, "... k"], ...] = torch.autograd.grad(
            target_out[..., feature_idx].sum(), list(post_acts.values()), retain_graph=True
        )
        for i, param_matrix_name in enumerate(post_acts.keys()):
            inner_acts = einops.einsum(
                pre_acts[param_matrix_name].detach().clone(),
                subnet_params[param_matrix_name],
                "... d_in, ... k d_in d_out -> ... k d_out",
            )
            layer_attribution_scores[i] += einops.einsum(
                grad_post_acts[i].detach(), inner_acts, "... d_out ,... k d_out -> ... k"
            ).pow(2)

    return layer_attribution_scores


def collect_subnetwork_attributions(
    spd_model: SPDModel | SPDFullRankModel | SPDRankPenaltyModel,
    target_model: Model,
    device: str,
    n_instances: int | None = None,
) -> Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"]:
    """
    Collect subnetwork attributions.

    This function creates a test batch using an identity matrix, passes it through the model,
    and collects the attributions.

    Args:
        spd_model: The model to collect attributions on.
        target_model: The target model to collect attributions on.
        pre_acts: The activations after the parameter matrix in the target model.
        device: The device to run computations on.
        n_instances: The number of instances in the batch.

    Returns:
        The attribution scores.
    """
    test_batch = torch.eye(spd_model.n_features, device=device)
    if n_instances is not None:
        test_batch = einops.repeat(
            test_batch, "batch n_features -> batch n_instances n_features", n_instances=n_instances
        )

    target_out, pre_acts, post_acts = target_model(test_batch)
    attribution_scores = calc_grad_attributions(
        target_out=target_out,
        subnet_params=spd_model.all_subnetwork_params(),
        pre_acts=pre_acts,
        post_acts=post_acts,
        k=spd_model.k,
    )
    return attribution_scores


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
    target_out: Float[Tensor, "... n_features"],
    pre_acts: dict[str, Float[Tensor, "batch n_instances d_in"] | Float[Tensor, "batch d_in"]],
    post_acts: dict[str, Float[Tensor, "batch n_instances d_out"] | Float[Tensor, "batch d_out"]],
    inner_acts: dict[str, Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"]],
    attribution_type: Literal["ablation", "gradient", "activation"],
) -> Float[Tensor, "batch n_instances k"] | Float[Tensor, "batch k"]:
    attributions = None
    if attribution_type == "ablation":
        attributions = calc_ablation_attributions(model=model, batch=batch, out=out)
    elif attribution_type == "gradient":
        attributions = calc_grad_attributions(
            target_out=target_out,
            pre_acts=pre_acts,
            subnet_params=model.all_subnetwork_params(),
            post_acts=post_acts,
            k=model.k,
        )
    elif attribution_type == "activation":
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
        Float[Tensor, "batch d_model_out"] | Float[Tensor, "batch n_instances d_model_out"]
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
    target_model: Model,
    input_array: Float[Tensor, "batch n_inputs"],
    attribution_type: Literal["gradient", "ablation", "activation"],
    batch_topk: bool,
    topk: float,
    distil_from_target: bool,
    topk_mask: Float[Tensor, "batch k"] | Float[Tensor, "batch n_instances k"] | None = None,
) -> SPDOutputs:
    # non-SPD model, and SPD-model non-topk forward pass
    target_model_output, pre_acts, post_acts = target_model(input_array)

    model_output_spd, layer_acts, inner_acts = spd_model(input_array)
    attribution_scores = calculate_attributions(
        model=spd_model,
        batch=input_array,
        out=model_output_spd,
        target_out=target_model_output,
        pre_acts=pre_acts,
        post_acts=post_acts,
        inner_acts=inner_acts,
        attribution_type=attribution_type,
    )

    if topk_mask is None:
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


DataGenerationType = Literal[
    "exactly_one_active",
    "exactly_two_active",
    "exactly_three_active",
    "exactly_four_active",
    "exactly_five_active",
    "at_least_zero_active",
]


class SparseFeatureDataset(
    Dataset[
        tuple[
            Float[Tensor, "batch n_instances n_features"],
            Float[Tensor, "batch n_instances n_features"],
        ]
    ]
):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
        data_generation_type: DataGenerationType = "at_least_zero_active",
        value_range: tuple[float, float] = (0.0, 1.0),
        synced_inputs: list[list[int]] | None = None,
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.data_generation_type = data_generation_type
        self.value_range = value_range
        self.synced_inputs = synced_inputs

    def __len__(self) -> int:
        return 2**31

    def sync_inputs(
        self, batch: Float[Tensor, "batch n_instances n_features"]
    ) -> Float[Tensor, "batch n_instances n_features"]:
        assert self.synced_inputs is not None
        all_indices = [item for sublist in self.synced_inputs for item in sublist]
        assert len(all_indices) == len(set(all_indices)), "Synced inputs must be non-overlapping"
        for indices in self.synced_inputs:
            mask = torch.zeros_like(batch, dtype=torch.bool)
            # First, get the samples for which there is a non-zero value for any of the indices
            non_zero_samples = (batch[..., indices] != 0.0).any(dim=-1)
            for idx in indices:
                mask[..., idx] = non_zero_samples
            # Now generate random values in value_range and apply them to the masked elements
            max_val, min_val = self.value_range
            random_values = torch.rand(
                batch.shape[0], self.n_instances, self.n_features, device=self.device
            )
            random_values = random_values * (max_val - min_val) + min_val
            batch = torch.where(mask, random_values, batch)
        return batch

    def generate_batch(
        self, batch_size: int
    ) -> tuple[
        Float[Tensor, "batch n_instances n_features"], Float[Tensor, "batch n_instances n_features"]
    ]:
        # TODO: This is a hack to keep backward compatibility. Probably best to have
        # data_generation_type: Literal["exactly_n_active", "at_least_zero_active"] and
        # data_generation_n: PositiveInt
        number_map = {
            "exactly_one_active": 1,
            "exactly_two_active": 2,
            "exactly_three_active": 3,
            "exactly_four_active": 4,
            "exactly_five_active": 5,
        }
        if self.data_generation_type in number_map:
            n = number_map[self.data_generation_type]
            batch = self._generate_n_feature_active_batch(batch_size, n=n)
        elif self.data_generation_type == "at_least_zero_active":
            batch = self._generate_multi_feature_batch(batch_size)
            if self.synced_inputs is not None:
                batch = self.sync_inputs(batch)
        else:
            raise ValueError(f"Invalid generation type: {self.data_generation_type}")

        return batch, batch.clone().detach()

    def _generate_n_feature_active_batch(
        self, batch_size: int, n: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch with exactly n features active per sample and instance.

        Args:
            batch_size: Number of samples in the batch
            n: Number of features to activate per sample and instance
        """
        if n > self.n_features:
            raise ValueError(
                f"Cannot activate {n} features when only {self.n_features} features exist"
            )

        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)

        # Create indices for all features
        feature_indices = torch.arange(self.n_features, device=self.device)
        # Expand to batch size and n_instances
        feature_indices = feature_indices.expand(batch_size, self.n_instances, self.n_features)

        # For each instance in the batch, randomly permute the features
        perm = torch.rand_like(feature_indices.float()).argsort(dim=-1)
        permuted_features = feature_indices.gather(dim=-1, index=perm)

        # Take first n indices for each instance - guaranteed no duplicates
        active_features = permuted_features[..., :n]

        # Generate random values in value_range for the active features
        min_val, max_val = self.value_range
        random_values = torch.rand(batch_size, self.n_instances, n, device=self.device)
        random_values = random_values * (max_val - min_val) + min_val

        # Place each active feature
        for i in range(n):
            batch.scatter_(
                dim=2, index=active_features[..., i : i + 1], src=random_values[..., i : i + 1]
            )

        return batch

    def _masked_batch_generator(
        self, total_batch_size: int
    ) -> Float[Tensor, "total_batch_size n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.

        Args:
            total_batch_size: Number of samples in the batch (either `batch_size` or
                `batch_size * n_instances`)
        """
        min_val, max_val = self.value_range
        batch = (
            torch.rand((total_batch_size, self.n_features), device=self.device)
            * (max_val - min_val)
            + min_val
        )
        mask = torch.rand_like(batch) < self.feature_probability
        return batch * mask

    def _generate_multi_feature_batch(
        self, batch_size: int
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`."""
        total_batch_size = batch_size * self.n_instances
        batch = self._masked_batch_generator(total_batch_size)
        return einops.rearrange(
            batch,
            "(batch n_instances) n_features -> batch n_instances n_features",
            batch=batch_size,
        )

    def _generate_multi_feature_batch_no_zero_samples(
        self, batch_size: int, buffer_ratio: float
    ) -> Float[Tensor, "batch n_instances n_features"]:
        """Generate a batch where each feature activates independently with probability
        `feature_probability`.

        Ensures that there are no zero samples in the batch.

        Args:
            batch_size: Number of samples in the batch
            buffer_ratio: First generate `buffer_ratio * total_batch_size` samples and count the
                number of samples with all zeros. Then generate another `buffer_ratio *
                n_zeros` samples and fill in the zero samples. Continue until there are no zero
                samples.
        """
        total_batch_size = batch_size * self.n_instances
        buffer_size = int(total_batch_size * buffer_ratio)
        batch = torch.empty(0, device=self.device, dtype=torch.float32)
        n_samples_needed = total_batch_size
        while True:
            buffer = self._masked_batch_generator(buffer_size)
            # Get the indices of the non-zero samples in the buffer
            valid_indices = buffer.sum(dim=-1) != 0
            batch = torch.cat((batch, buffer[valid_indices][:n_samples_needed]))
            if len(batch) == total_batch_size:
                break
            else:
                # We don't have enough valid samples
                n_samples_needed = total_batch_size - len(batch)
                buffer_size = int(n_samples_needed * buffer_ratio)
        return einops.rearrange(
            batch,
            "(batch n_instances) n_features -> batch n_instances n_features",
            batch=batch_size,
        )


def compute_feature_importances(
    batch_size: int,
    n_instances: int,
    n_features: int,
    importance_val: float | None,
    device: str,
) -> Float[Tensor, "batch_size n_instances n_features"]:
    # Defines a tensor where the i^th feature has importance importance^i
    if importance_val is None or importance_val == 1.0:
        importance_tensor = torch.ones(batch_size, n_instances, n_features, device=device)
    else:
        powers = torch.arange(n_features, device=device)
        importances = torch.pow(importance_val, powers)
        # Now make it a tensor of shape (batch_size, n_instances, n_features)
        importance_tensor = einops.repeat(
            importances,
            "n_features -> batch_size n_instances n_features",
            batch_size=batch_size,
            n_instances=n_instances,
        )
    return importance_tensor


def calc_recon_mse(
    output: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    labels: Float[Tensor, "batch n_features"] | Float[Tensor, "batch n_instances n_features"],
    has_instance_dim: bool = False,
) -> Float[Tensor, ""] | Float[Tensor, " n_instances"]:
    recon_loss = (output - labels) ** 2
    if recon_loss.ndim == 3:
        assert has_instance_dim
        recon_loss = einops.reduce(recon_loss, "b i f -> i", "mean")
    elif recon_loss.ndim == 2:
        recon_loss = recon_loss.mean()
    else:
        raise ValueError(f"Expected 2 or 3 dims in recon_loss, got {recon_loss.ndim}")
    return recon_loss
