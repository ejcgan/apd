import torch
from jaxtyping import Float
from torch import Tensor


def permute_to_identity(x: torch.Tensor, normalize_rows: bool = False) -> torch.Tensor:
    """Permute the rows of a matrix such that the maximum value in each column is on the leading
    diagonal.

    Args:
        x: The input matrix.
        normalize_rows: Whether to normalize the rows of the output matrix.
    """

    # Assert that arr only has two dimensions and that it is square
    assert x.dim() == 2
    assert x.shape[0] == x.shape[1]

    # Get the number of rows and columns
    n_rows, n_cols = x.shape

    # Find the row index of the maximum value in each column
    max_row_indices_raw = torch.argmax(x, dim=0)

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
