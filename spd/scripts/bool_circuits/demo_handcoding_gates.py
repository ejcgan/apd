"""Demonstrates exact handcoding of AND, OR, and NOT gates is possible using matrices."""
import torch
import torch.nn.functional as F
from torch import Tensor


def and_gate(x: Tensor) -> Tensor:
    W = torch.tensor([1.0, 1.0])
    b = torch.tensor(-1.0)
    return F.relu(torch.dot(W, x) + b)


print("AND")
print(and_gate(torch.tensor([0.0, 0.0])))
print(and_gate(torch.tensor([0.0, 1.0])))
print(and_gate(torch.tensor([1.0, 0.0])))
print(and_gate(torch.tensor([1.0, 1.0])))


def not_gate(x: Tensor) -> Tensor:
    W = torch.tensor([-1.0])
    b = torch.tensor(1)
    return torch.dot(W, x) + b


print("NOT")
print(not_gate(torch.tensor([0.0])))
print(not_gate(torch.tensor([1.0])))


def or_gate(x: Tensor) -> Tensor:
    W = torch.tensor([1.0, 1.0])
    return torch.dot(W, x) - and_gate(x)


print("OR")
print(or_gate(torch.tensor([0.0, 0.0])))
print(or_gate(torch.tensor([0.0, 1.0])))
print(or_gate(torch.tensor([1.0, 0.0])))
print(or_gate(torch.tensor([1.0, 1.0])))
