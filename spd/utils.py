import torch
from jaxtyping import Float
from torch import Tensor


def permute_to_identity(arr: torch.Tensor, normalize_rows: bool = False) -> torch.Tensor:
    """Permute the rows of a matrix such that the maximum value in each column is on the leading
    diagonal.

    Args:
        arr: The input matrix.
        normalize_rows: Whether to normalize the rows of the output matrix.
    """

    # First take the absolute value of the input matrix
    x = torch.abs(arr)

    # Assert that arr only has two dimensions
    assert arr.dim() == 2

    # Get the number of rows and columns
    n_rows, n_cols = x.shape

    # Find the row index of the maximum value in each column
    max_row_indices_raw = torch.argmax(x, dim=0)

    # Remove duplicates from the max_row_indices and preserve the order
    max_row_indices = []
    for i in max_row_indices_raw:
        if i not in max_row_indices:
            max_row_indices.append(i)
    max_row_indices = torch.tensor(max_row_indices)

    remaining_indices = torch.tensor([i for i in range(n_rows) if i not in max_row_indices])

    out_rows = torch.zeros_like(x)

    # If there are more columns than rows, stop assigning rows to the output matrix after n_rows
    if n_cols > n_rows:
        max_row_indices = max_row_indices[:n_rows]
        remaining_indices = torch.tensor([])
    assert len(max_row_indices) + len(remaining_indices) == n_rows
    assert len(max_row_indices) == len(torch.unique(max_row_indices)), "Unusual matrix found"

    for i in range(len(max_row_indices)):
        out_rows[i] = x[max_row_indices[i]]
    for i in range(len(remaining_indices)):
        out_rows[i + len(max_row_indices)] = x[remaining_indices[i]]

    if normalize_rows:
        out_rows = out_rows / out_rows.norm(dim=1, p=2, keepdim=True)
    return out_rows


def calculate_closeness_to_identity(x: Float[Tensor, "... a b"]) -> float:
    """Frobenius norm of the difference between the input matrix and the identity matrix.

    If x has more than two dimensions, the result is meaned over all but the final two dimensions.
    """
    eye = torch.eye(n=x.shape[-2], m=x.shape[-1], device=x.device)
    return torch.norm(x - eye, p="fro", dim=(-2, -1)).mean().item()
