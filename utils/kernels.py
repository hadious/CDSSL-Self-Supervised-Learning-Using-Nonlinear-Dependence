import torch 
from typing import Optional


def rbf_kernel_fast(z: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    device = z.device
    z = z.to(device)
    pairwise_distances = torch.cdist(z, z, p=2).pow(2)
    if gamma is None:
        gamma = 1.0 / z.size(1)
    kernel_matrix = torch.exp(-gamma * pairwise_distances)

    return kernel_matrix

def rbf_kernel_fast_cross(z1: torch.Tensor, z2: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    device = z1.device
    z1, z2 = z1.to(device), z2.to(device)
    pairwise_distances = torch.cdist(z1, z2, p=2).pow(2)
    if gamma is None:
        gamma = 1.0 / z1.size(1)
    kernel_matrix = torch.exp(-gamma * pairwise_distances)
    return kernel_matrix

def linear_kernel_cross(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return z1 @ z2.T 

def polynomial_kernel_cross(z1: torch.Tensor, z2: torch.Tensor, degree: int = 3, c: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    linear_term = z1 @ z2.T
    return (alpha * linear_term + c).pow(degree)

def sigmoid_kernel(z: torch.Tensor, alpha: float = 1.0, c: float = 0.0) -> torch.Tensor:
    linear_term = z @ z.T
    return torch.tanh(alpha * linear_term + c)

def sigmoid_kernel_cross(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 1.0, c: float = 0.0) -> torch.Tensor:
    linear_term = z1 @ z2.T
    return torch.tanh(alpha * linear_term + c)

def polynomial_kernel(z: torch.Tensor, degree: int = 3, c: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    linear_term = z @ z.T
    return (alpha * linear_term + c).pow(degree)

def rbf_kernel_matrix(X, gamma=None):
    pairwise_sq_dists = torch.cdist(X, X, p=2).pow(2)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = torch.exp(-gamma * pairwise_sq_dists)
    return K