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
        list[Float[Tensor, "... d_layer_out"]],  # layer activations
        list[Float[Tensor, "... k"]],  # inner activations
    ]:
        pass

    @abstractmethod
    def forward_topk(
        self, x: Float[Tensor, "... d_model_in"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        list[Float[Tensor, "... d_layer_out"]],  # layer activations
        list[Float[Tensor, "... k"]],  # inner activations
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDModel":
        pass

    @abstractmethod
    def all_As(self) -> list[Float[Tensor, "... d_layer_in k"]]:
        """Pre-normalized A matrices."""
        pass

    @abstractmethod
    def all_Bs(self) -> list[Float[Tensor, "... k d_layer_out"]]:
        pass

    @abstractmethod
    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        pass

    @abstractmethod
    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        pass


class SPDFullRankModel(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, x: Float[Tensor, "... d_model_in"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        list[Float[Tensor, "... d_layer_out"]],  # layer activations
        list[Float[Tensor, "... k d_layer_out"]],  # inner activations
    ]:
        pass

    @abstractmethod
    def forward_topk(
        self, x: Float[Tensor, "... d_model_in"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... d_model_out"],  # output
        list[Float[Tensor, "... d_layer_out"]],  # layer activations
        list[Float[Tensor, "... k d_layer_out"]],  # inner activations
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDFullRankModel":
        pass

    @abstractmethod
    def all_subnetwork_params(self) -> list[Float[Tensor, "... k d_layer_in d_layer_out"]]:
        pass


class Model(ABC, nn.Module):
    @abstractmethod
    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        pass
