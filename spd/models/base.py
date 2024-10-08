from abc import ABC, abstractmethod
from pathlib import Path

from jaxtyping import Bool, Float
from torch import Tensor, nn


class SPDModel(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, x: Float[Tensor, "... d_model_in"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        dict[str, Float[Tensor, "... d_layer_out"]],  # layer activations
        dict[str, Float[Tensor, "... k"]],  # inner activations
    ]:
        pass

    @abstractmethod
    def forward_topk(
        self, x: Float[Tensor, "... d_model_in"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        dict[str, Float[Tensor, "... d_layer_out"]],  # layer activations
        dict[str, Float[Tensor, "... k"]],  # inner activations
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDModel":
        pass

    @abstractmethod
    def all_As_and_Bs(
        self,
    ) -> dict[str, tuple[Float[Tensor, "... d_layer_in k"], Float[Tensor, "... k d_layer_out"]]]:
        """Dict of tuples containing A and B matrices, keyed by the layer name."""
        pass

    @abstractmethod
    def all_subnetwork_params_summed(self) -> dict[str, Float[Tensor, "d_layer_in d_layer_out"]]:
        pass

    @abstractmethod
    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        pass

    @abstractmethod
    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        pass

    @abstractmethod
    def set_subnet_to_zero(self, subnet_idx: int) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def restore_subnet(self, subnet_idx: int, stored_vals: dict[str, Tensor]) -> None:
        pass


class SPDFullRankModel(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, x: Float[Tensor, "... d_model_in"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        dict[str, Float[Tensor, "... d_layer_out"]],  # layer activations
        dict[str, Float[Tensor, "... k d_layer_out"]],  # inner activations
    ]:
        pass

    @abstractmethod
    def forward_topk(
        self, x: Float[Tensor, "... d_model_in"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        dict[str, Float[Tensor, "... d_layer_out"]],  # layer activations
        dict[str, Float[Tensor, "... k d_layer_out"]],  # inner activations
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDFullRankModel":
        pass

    @abstractmethod
    def all_subnetwork_params(self) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def set_subnet_to_zero(self, subnet_idx: int) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def restore_subnet(self, subnet_idx: int, stored_vals: dict[str, Tensor]) -> None:
        pass


class Model(ABC, nn.Module):
    @abstractmethod
    def all_decomposable_params(self) -> dict[str, Tensor]:
        pass
