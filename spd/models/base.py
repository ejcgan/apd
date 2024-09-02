from abc import ABC, abstractmethod
from pathlib import Path

from jaxtyping import Bool, Float
from torch import Tensor, nn


class SPDModel(ABC, nn.Module):
    @abstractmethod
    def forward_topk(
        self, x: Float[Tensor, "... dim"], topk_mask: Bool[Tensor, "... k"]
    ) -> tuple[
        Float[Tensor, "... dim"],
        list[Float[Tensor, "... dim"]],
        list[Float[Tensor, "... k"]],
    ]:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str | Path) -> "SPDModel":
        pass

    @abstractmethod
    def all_As(self) -> list[Float[Tensor, "dim k"]]:
        """Pre-normalized A matrices."""
        pass

    @abstractmethod
    def all_Bs(self) -> list[Float[Tensor, "k dim"]]:
        pass

    @abstractmethod
    def set_matrices_to_unit_norm(self) -> None:
        """Set the matrices that need to be normalized to unit norm."""
        pass

    @abstractmethod
    def fix_normalized_adam_gradients(self) -> None:
        """Modify the gradient by subtracting it's component parallel to the activation."""
        pass


class Model(ABC, nn.Module):
    @abstractmethod
    def all_decomposable_params(self) -> list[Float[Tensor, "..."]]:
        pass
