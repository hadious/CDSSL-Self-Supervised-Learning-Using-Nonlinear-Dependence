import torch
import torch.nn.functional as F
from utils import kernels
from typing import Optional


def cross_correlation_between_samples(embedding1: torch.tensor, embedding2: torch.tensor, lambda_param: float) -> torch.tensor:
    """
    Cross correlation loss between samples.
    NOTE: it is equivalent to the loss function of Barlow. 
    NOTE: 
        - it is used for alignment, across augmentations, (i.e., invariance to the augmentation).
        - it is used for decorrelating different samples, across augmentations, and avoiding collapse.
    NOTE: this should be minimized, so that:
        - maximize correlation between z_i and z'_i to one
        - minimize correlation between z_i and z'_j for i != j
    """
    embedding1 = (embedding1 - embedding1.mean(dim=0)) / embedding1.std(dim=0)
    embedding2 = (embedding2 - embedding2.mean(dim=0)) / embedding2.std(dim=0)
    batch_size = embedding1.size(0)
    cross_correlation = torch.mm(embedding1.T, embedding2) / batch_size  
    diag_loss = torch.sum((torch.diag(cross_correlation) - 1) ** 2)
    off_diag_loss = torch.sum(cross_correlation**2) - torch.sum(torch.diag(cross_correlation) ** 2)
    loss = diag_loss + lambda_param * off_diag_loss
    return loss

def cross_correlation_between_features(embedding1: torch.tensor, embedding2: torch.tensor, lambda_param: float) -> torch.tensor:
    """
    Cross correlation loss between features.
    NOTE: it is equivalent to the variance and covariance terms in VICReg.
    NOTE: 
        - it is used for decorrelating different features, across augmentations, and avoiding collapse.
    NOTE: this should be minimized, so that:
        - maximize correlation between z^i and z'^i to one
        - minimize correlation between z^i and z'^j for i != j
    """
    loss = cross_correlation_between_samples(embedding1=embedding1.T, embedding2=embedding2.T, lambda_param=lambda_param)
    return loss

def auto_correlation_between_samples(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
    """
    Auto correlation loss between samples.
    NOTE: 
        - it is used for decorrelating different samples, within batch, and avoiding collapse.
    NOTE: this should be minimized, so that:
        - correlation between z_i and z_i is always one already!
        - minimize correlation between z_i and z_j for i != j
    """
    def _loss(embedding: torch.tensor) -> torch.tensor:
        embedding = (embedding - embedding.mean(dim=0)) / embedding.std(dim=0)
        batch_size = embedding.size(0)
        auto_correlation = torch.mm(embedding.T, embedding) / batch_size
        off_diag_loss = torch.sum(auto_correlation**2) - torch.sum(torch.diag(auto_correlation) ** 2)
        loss = off_diag_loss  # NOTE: diag_loss is always one
        return loss
    loss_total = _loss(embedding=embedding1) + _loss(embedding=embedding2)
    return loss_total

def auto_correlation_between_features(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
    """
    Auto correlation loss between features.
    NOTE: 
        - it is used for decorrelating different features, within batch, and avoiding collapse.
    NOTE: this should be minimized, so that:
        - correlation between z^i and z^i is always one already!
        - minimize correlation between z^i and z^j for i != j
    """
    loss = auto_correlation_between_samples(embedding1=embedding1.T, embedding2=embedding2.T)
    return loss

def cross_dependence_between_samples(embedding1: torch.tensor, embedding2: torch.tensor, method: Optional[str] = "HSIC") -> torch.tensor:
    """
    Cross dependence loss between samples.
    NOTE: 
        - it is used for alignment, across augmentations, (i.e., invariance to the augmentation).
    NOTE: this should be minimized, so that:
        - maximize dependence between z_i and z'_i
    """
    if method == "HSIC":
        loss_between_sample_i_in_z_and_sample_j_in_z_prime = _hsic_between_corresponding_batches(embedding1=embedding1, embedding2=embedding2, scramble_embedding2=False)
        loss = -1 * loss_between_sample_i_in_z_and_sample_j_in_z_prime
    elif method == "MMD":
        loss_between_sample_i_in_z_and_sample_j_in_z_prime = _mmd_loss(embedding1=embedding1, embedding2=embedding2, scramble_embedding2=False)
        loss = loss_between_sample_i_in_z_and_sample_j_in_z_prime
    return loss

def cross_dependence_between_features(embedding1: torch.tensor, embedding2: torch.tensor, method: Optional[str] = "HSIC") -> torch.tensor:
    """
    Cross dependence loss between features.
    NOTE: 
        - it is used for reducing the dependency, across augmentations, and avoiding collapse.
    NOTE: this should be minimized, so that:
        - maximize dependence between z^i and z'^i
        - minimize dependence (redundancy of different features) between z^i and z'^j for i != j
    """
    if method == "HSIC":
        loss_between_feature_i_in_z_and_feature_i_in_z_prime = _hsic_between_corresponding_batches(embedding1=embedding1.T, embedding2=embedding2.T, scramble_embedding2=False)
        loss_between_feature_i_in_z_and_feature_j_in_z_prime = _hsic_between_corresponding_batches(embedding1=embedding1.T, embedding2=embedding2.T, scramble_embedding2=True)
        loss = (-1 * loss_between_feature_i_in_z_and_feature_i_in_z_prime) + (loss_between_feature_i_in_z_and_feature_j_in_z_prime)
    elif method == "MMD":
        loss_between_feature_i_in_z_and_feature_i_in_z_prime = _mmd_loss(embedding1=embedding1.T, embedding2=embedding2.T, scramble_embedding2=False)
        loss_between_feature_i_in_z_and_feature_j_in_z_prime = _mmd_loss(embedding1=embedding1.T, embedding2=embedding2.T, scramble_embedding2=True)
        loss = (loss_between_feature_i_in_z_and_feature_i_in_z_prime) + (-1 * loss_between_feature_i_in_z_and_feature_j_in_z_prime)
    return loss

def auto_dependence_between_samples(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
    """
    Auto dependence loss between samples.
    NOTE: it is used for maximizing diversity of samples (in RKHS), within batch.
    NOTE: this should be minimized, so that:
        - maximize diversity (complexity) (variablity) of z_i's within batch
    """
    # if method == "HSIC":
    loss_between_samples_in_z = _hsic_between_corresponding_batches(embedding1=embedding1, embedding2=embedding1, scramble_embedding2=False)
    loss_between_samples_in_z_prime = _hsic_between_corresponding_batches(embedding1=embedding2, embedding2=embedding2, scramble_embedding2=False)
    loss = (-1 * loss_between_samples_in_z) + (-1 * loss_between_samples_in_z_prime)
    # elif method == "MMD":
    #     loss_between_samples_in_z = _mmd_loss(embedding1=embedding1, embedding2=embedding1, scramble_embedding2=False)
    #     loss_between_samples_in_z_prime = _mmd_loss(embedding1=embedding2, embedding2=embedding2, scramble_embedding2=False)
    #     loss = (loss_between_samples_in_z) + (loss_between_samples_in_z_prime)
    return loss

def auto_dependence_between_features(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
    """
    Auto dependence loss between features.
    NOTE: it is used for maximizing diversity of features (reducing redundancy of features) (in RKHS), within batch.
    NOTE: this should be minimized, so that:
        - maximize diversity (complexity) (variablity) of z^i's within batch -> e.g., nose should be different from mouth
    """
    # if method == "HSIC":
    loss_between_samples_in_z = _hsic_between_corresponding_batches(embedding1=embedding1.T, embedding2=embedding1.T, scramble_embedding2=False)
    loss_between_samples_in_z_prime = _hsic_between_corresponding_batches(embedding1=embedding2.T, embedding2=embedding2.T, scramble_embedding2=False)
    loss = (-1 * loss_between_samples_in_z) + (-1 * loss_between_samples_in_z_prime)
    # elif method == "MMD":
    #     loss_between_samples_in_z = _mmd_loss(embedding1=embedding1.T, embedding2=embedding1.T, scramble_embedding2=False)
    #     loss_between_samples_in_z_prime = _mmd_loss(embedding1=embedding2.T, embedding2=embedding2.T, scramble_embedding2=False)
    #     loss = (loss_between_samples_in_z) + (loss_between_samples_in_z_prime)
    return loss

def _hsic_between_corresponding_batches(embedding1: torch.Tensor, embedding2: torch.Tensor, scramble_embedding2: Optional[bool] = False) -> torch.Tensor:
    batch_size, feature_dim = embedding1.size()
    H = torch.eye(batch_size, device=embedding1.device) - (1 / batch_size) * torch.ones(batch_size, batch_size, device=embedding1.device)
 
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)

    if scramble_embedding2:
        # Scramble rows
        permutation = torch.randperm(embedding2.size(0))  # Generate a random permutation of row indices
        embedding2 = embedding2[permutation]            # Apply the permutation to the rows

    K_embedding1 = kernels.rbf_kernel_fast(z=embedding1)
    K_embedding2 = kernels.rbf_kernel_fast(z=embedding2)
    hsic = torch.trace(K_embedding1 @ H @ K_embedding2 @ H)
    return hsic

def _mmd_loss(embedding1: torch.tensor, embedding2: torch.tensor, scramble_embedding2: Optional[bool] = False, kernel_bandwidth: float = 1.0) -> torch.tensor:
    # Centering
    embedding1 = embedding1 - embedding1.mean(dim=0, keepdim=True)
    embedding2 = embedding2 - embedding2.mean(dim=0, keepdim=True)

    if scramble_embedding2:
        # Scramble rows
        permutation = torch.randperm(embedding2.size(0))  # Generate a random permutation of row indices
        embedding2 = embedding2[permutation]            # Apply the permutation to the rows

    # Compute squared pairwise distances
    XX = torch.cdist(embedding1, embedding1, p=2) ** 2
    YY = torch.cdist(embedding2, embedding2, p=2) ** 2
    XY = torch.cdist(embedding1, embedding2, p=2) ** 2
    
    # Apply the RBF kernel
    K_XX = torch.exp(-XX / (2 * kernel_bandwidth ** 2))
    K_YY = torch.exp(-YY / (2 * kernel_bandwidth ** 2))
    K_XY = torch.exp(-XY / (2 * kernel_bandwidth ** 2))
    
    # MMD loss
    loss = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return loss