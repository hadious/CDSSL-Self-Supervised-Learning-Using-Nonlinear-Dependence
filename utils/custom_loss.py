import torch 
import torch.nn.functional as F
from scipy import stats
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from utils import kernels
from torch.nn.functional import mse_loss, relu


def variance_loss(h, threshold=1.0, epsilon=1e-4):
    # std_dev = torch.sqrt(h.var(dim=0) + epsilon)
    # hinge_loss = torch.mean(torch.relu(threshold - std_dev))
    # return hinge_loss
    return relu(threshold - h.std(0)).mean()


def covariance_loss(z: torch.Tensor):
    z = z - z.mean(dim=0)
    batch_size = z.size(0)
    cov = (z.T @ z) / (batch_size - 1)
    cov_diag = torch.diagonal(cov)
    cov_loss = ((cov - torch.diag(cov_diag)) ** 2).sum() / z.size(1)
    return cov_loss

def invariance_loss(z1: torch.Tensor, z2: torch.Tensor):
    inv_loss = F.mse_loss(z1, z2)
    return inv_loss

def kernel_vicreg(z1: torch.Tensor, z2: torch.Tensor, weight_invariance: int, weight_variance: int, weight_covariance: int):
    def _double_center_the_kernel_matrix(kernel_matrix: torch.Tensor):
        n = kernel_matrix.shape[0]
        one_n = torch.ones((n, n), device=kernel_matrix.device) / n
        kernel_matrix_centered = kernel_matrix - (one_n @ kernel_matrix) - (kernel_matrix @ one_n) + (one_n @ kernel_matrix @ one_n)
        return kernel_matrix_centered

    def _invariance_loss(kernel_matrix_z1: torch.Tensor, kernel_matrix_z2: torch.Tensor, kernel_matrix_z1_z2: torch.Tensor):
        result = torch.sum(kernel_matrix_z1) - 2 * torch.sum(kernel_matrix_z1_z2) + torch.sum(kernel_matrix_z2)
        return result

    def _variance_loss(kernel_matrix_centered: torch.Tensor, threshold: float = 1.0, epsilon: float = 1e-4) -> torch.Tensor:
        igenvalues, eigenvectors = torch.linalg.eigh(kernel_matrix_centered)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        std_dev = torch.sqrt(eigenvalues + epsilon)
        hinge_loss = torch.mean(torch.relu(threshold - std_dev))
        return hinge_loss
    
    def _covariance_loss(kernel_matrix_centered: torch.Tensor) -> torch.Tensor:
        kernel_matrix_centered_diag = torch.diagonal(kernel_matrix_centered)
        cov_loss = ((kernel_matrix_centered - torch.diag(kernel_matrix_centered_diag)) ** 2).sum() / kernel_matrix_centered.size(1)
        return cov_loss
    
    # calculate kernels:
    kernel_matrix_z1 = kernels.rbf_kernel_fast(z=z1)
    kernel_matrix_z2 = kernels.rbf_kernel_fast(z=z2)
    kernel_matrix_z1_z2 = kernels.rbf_kernel_fast_cross(z1=z1, z2=z2)

    # double center the kernel matrix:
    kernel_matrix_z1_centered = _double_center_the_kernel_matrix(kernel_matrix=kernel_matrix_z1)
    kernel_matrix_z2_centered = _double_center_the_kernel_matrix(kernel_matrix=kernel_matrix_z2)
    kernel_matrix_z1_z2_centered = _double_center_the_kernel_matrix(kernel_matrix=kernel_matrix_z1_z2)

    # invariance (MSE) loss:
    invariance_loss_value_z1_z2 = _invariance_loss(kernel_matrix_z1=kernel_matrix_z1_centered, kernel_matrix_z2=kernel_matrix_z2_centered, kernel_matrix_z1_z2=kernel_matrix_z1_z2_centered)

    # variance loss:
    variance_loss_value_z1 = _variance_loss(kernel_matrix_centered=kernel_matrix_z1_centered, threshold=1.0, epsilon=1e-4)
    variance_loss_value_z2 = _variance_loss(kernel_matrix_centered=kernel_matrix_z2_centered, threshold=1.0, epsilon=1e-4)

    # covariance loss:
    covariance_loss_value_z1 = _covariance_loss(kernel_matrix_centered=kernel_matrix_z1_centered)
    covariance_loss_value_z2 = _covariance_loss(kernel_matrix_centered=kernel_matrix_z2_centered)

    # total loss:
    loss = (weight_invariance * invariance_loss_value_z1_z2) + (weight_variance * (variance_loss_value_z1 + variance_loss_value_z2)) + (weight_covariance * (covariance_loss_value_z1 + covariance_loss_value_z2))
    return loss

# KL Divergence function (vectorized version)
def kl_divergence_vectorized(p, q):
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=0)

# Vectorized centered pairwise JSD loss function
def centered_pairwise_jsd_loss_vectorized(features):
    batch_size, num_features = features.size()

    # Mean-center the features along the batch dimension
    features_centered = features - features.mean(dim=0, keepdim=True)

    # Apply softmax to each feature dimension across the batch
    p = F.softmax(features_centered, dim=0).unsqueeze(2)  # Shape: (batch_size, num_features, 1)
    q = p.transpose(1, 2)  # Shape: (batch_size, 1, num_features)

    # Compute the mean distribution for each pair
    m = 0.5 * (p + q)

    # Calculate JSD for all pairs at once
    jsd = 0.5 * (kl_divergence_vectorized(p, m) + kl_divergence_vectorized(q, m))

    # Mask out the diagonal (self-comparisons)
    mask = ~torch.eye(num_features, dtype=bool, device=features.device)
    jsd = jsd[mask].mean()  # Only average over non-diagonal pairs

    return jsd

def centered_pairwise_jsd_loss(z1, z2):
    l1  = centered_pairwise_jsd_loss_vectorized(z1)
    l2 = centered_pairwise_jsd_loss_vectorized(z2)
    return l1+l2

def get_high_cov_indices(cov):
    
    dim = cov.size(0)
    upper_tri_indices = torch.triu_indices(dim, dim, offset=1).to(cov.device)  # offset=1 to exclude the diagonal
    upper_tri_values = cov[upper_tri_indices[0], upper_tri_indices[1]]
    
    # select those upper than 13.6% (normal theory)
    mu, std = torch.mean(upper_tri_values), torch.std(upper_tri_values)
    threshold = mu + 2*std

    high_cov_mask = (upper_tri_values > threshold).to(cov.device)
    high_cov_indices = upper_tri_indices[:, high_cov_mask]

    return high_cov_indices

def pairwise_hsic_loss_selected_pairs(h, high_cov_indices):
    N, d = h.size()  # N: batch size, d: number of features (dimensions)

    # Compute the RBF kernel matrix for each feature
    K = torch.zeros((N, N, d), device=h.device)  # Kernel matrix for each feature
    for i in range(d):
        h_i = h[:, i].unsqueeze(1)
        K[:, :, i] = kernels.rbf_kernel_matrix(h_i)

    # Centering matrix H
    H = torch.eye(N).to(h.device) - (1.0 / N) * torch.ones(N, N).to(h.device)
    # print (h.device)
    # Apply centering using broadcasting and matrix multiplication
    K_centered = torch.matmul(H, K.permute(2, 0, 1))  # Center each kernel
    K_centered = torch.matmul(K_centered, H)  # Center again from the right
    K_centered = K_centered.permute(1, 2, 0)  # Permute back

    # Compute the pairwise HSIC matrix by calculating the covariance for the selected pairs
    hsic_loss = 0.0
    for (i, j) in zip(high_cov_indices[0], high_cov_indices[1]):
        # Calculate covariance for the selected feature pairs (i, j)
        K_i = K_centered[:, :, i]
        K_j = K_centered[:, :, j]
        cov_hsic_ij = torch.sum(K_i * K_j) / (N - 1)**2
        hsic_loss += cov_hsic_ij

    # Normalize by the number of selected feature pairs
    total_pairs = high_cov_indices[0].size(0)
    hsic_loss /= total_pairs

    return hsic_loss

def hsic_loss_selected_paris(z1, z2):
    z1 = z1 - z1.mean(dim=0)
    batch_size = z1.size(0)
    cov1 = (z1.T @ z1) / (batch_size - 1)
    index_pairs1 = get_high_cov_indices(cov1)
    loss_z1 = pairwise_hsic_loss_selected_pairs(z1,index_pairs1)
    z2 = z2 - z2.mean(dim=0)
    cov2 = (z2.T @ z2) / (batch_size - 1)
    index_pairs2 = get_high_cov_indices(cov2)
    loss_z2 = pairwise_hsic_loss_selected_pairs(z2, index_pairs2)
    return loss_z1 + loss_z2

def mmd_loss(X, Y, kernel_bandwidth=1.0):

    # Centering
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute squared pairwise distances
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    
    # Apply the RBF kernel
    K_XX = torch.exp(-XX / (2 * kernel_bandwidth ** 2))
    K_YY = torch.exp(-YY / (2 * kernel_bandwidth ** 2))
    K_XY = torch.exp(-XY / (2 * kernel_bandwidth ** 2))
    
    # MMD loss
    loss = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return loss

def hsic_loss(features, sigma=1.0):
 
    batch_size, feature_dim = features.size()
    H = torch.eye(batch_size, device=features.device) - (1 / batch_size) * torch.ones(batch_size, batch_size, device=features.device)
    hsic_loss_value = 0.0
    kernel_matrices = []
    for i in range(feature_dim):
        feature_i = features[:, i].unsqueeze(1) 
        dist_matrix = torch.cdist(feature_i, feature_i, p=2) ** 2
        kernel_i = torch.exp(-dist_matrix / (2 * sigma ** 2))
        # import pdb 
        # pdb.set_trace()
        kernel_matrices.append(kernel_i)

    K_sum = sum(kernel_matrices)
 
    for j in range(feature_dim):
        K_j = kernel_matrices[j]   
        K_others = K_sum - K_j  
        hsic_j = torch.trace(K_j @ H @ K_others @ H) / (batch_size - 1)
        hsic_loss_value += hsic_j

    return hsic_loss_value / feature_dim  

def hsic_loss_selected_pairs_new(features, sigma=1.0):
    
    batch_size, feature_dim = features.size()

    H = torch.eye(batch_size, device=features.device) - (1 / batch_size) * torch.ones(batch_size, batch_size, device=features.device)

    covariance_matrix = torch.cov(features.T)  
    
    upper_triangle = covariance_matrix[torch.triu(torch.ones_like(covariance_matrix), diagonal=1).bool()]
    mean, std = upper_triangle.mean(), upper_triangle.std()

    threshold = mean + 7* std  
    # high_cov_indices = (upper_triangle > threshold).nonzero(as_tuple=False)

    high_cov_indices = (torch.abs(covariance_matrix) > threshold).nonzero(as_tuple=False) 

    pair_number = 1

    hsic_loss_value = 0.0
    for i, j in high_cov_indices:
        if i >= j:  # Avoid redundant or diagonal computations
            continue
        
        pair_number += 1


        feature_i = features[:, i].unsqueeze(1)
        dist_matrix_i = torch.cdist(feature_i, feature_i, p=2) ** 2
        kernel_i = torch.exp(-dist_matrix_i / (2 * sigma ** 2))


        feature_j = features[:, j].unsqueeze(1)
        dist_matrix_j = torch.cdist(feature_j, feature_j, p=2) ** 2
        kernel_j = torch.exp(-dist_matrix_j / (2 * sigma ** 2))

        hsic_ij = torch.trace(kernel_i @ H @ kernel_j @ H) / (batch_size - 1)
        hsic_loss_value += hsic_ij

    return torch.tensor(hsic_loss_value / pair_number, device=features.device)

def hsic_between_corresponding_batches(z1: torch.Tensor, z2: torch.Tensor, scramble_z2: bool) -> torch.Tensor:
    batch_size, feature_dim = z1.size()
    H = torch.eye(batch_size, device=z1.device) - (1 / batch_size) * torch.ones(batch_size, batch_size, device=z1.device)
 
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    if scramble_z2:
        # Scramble rows
        permutation = torch.randperm(z2.size(0))  # Generate a random permutation of row indices
        z2 = z2[permutation]            # Apply the permutation to the rows

    K_z1 = kernels.rbf_kernel_fast(z=z1)
    K_z2 = kernels.rbf_kernel_fast(z=z2)
    hsic = torch.trace(K_z1 @ H @ K_z2 @ H)
    return hsic

def scatter_loss_feature(z1, z2):
    assert z1.shape == z2.shape, "z1 and z2 must have the same shape" 
    z = torch.cat([z1, z2], dim=0)   
    mean = torch.mean(z, dim=0, keepdim=True)   
    centered = z - mean  
    scatter_matrix = centered.T @ centered / (2 * z.size(0))   
    loss = torch.norm(scatter_matrix, p='fro') ** 2
    return loss

def coross_corolation(embedding1, embedding2, lambda_param):
    embedding1 = (embedding1 - embedding1.mean(0)) / embedding1.std(0)
    embedding2 = (embedding2 - embedding2.mean(0)) / embedding2.std(0)
    batch_size = embedding1.size(0)
    cross_correlation = torch.mm(embedding1.T, embedding2) / batch_size  
    diag_loss = torch.sum((torch.diag(cross_correlation) - 1) ** 2)
    off_diag_loss = torch.sum(cross_correlation**2) - torch.sum(torch.diag(cross_correlation) ** 2)
    loss = diag_loss + lambda_param * off_diag_loss
    # import pdb;pdb.set_trace()
    return loss

def entropy_loss(z, threshold=0.5):
    probs = F.softmax(z, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()
    
    # Hinge loss to enforce a minimum threshold
    loss_entropy = torch.relu(threshold - entropy)
    return loss_entropy

def orthogonalization_loss(z):
    z_centered = z - z.mean(dim=0, keepdim=True)  
    covariance_matrix = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    
    #  Frobenius norm of (Cov - Identity)
    identity_matrix = torch.eye(covariance_matrix.size(0), device=z.device)
    loss_orth = torch.norm(covariance_matrix - identity_matrix, p='fro')
    return loss_orth

def combined_loss_ent_orthogonality(z, alpha=1.0, beta=1.0, entropy_threshold=0.001):
    loss_var = entropy_loss(z, threshold=entropy_threshold)
    loss_cov = orthogonalization_loss(z)
    return alpha * loss_var + beta * loss_cov

def auto_corrolation(z):
    # Compute the mean for each column
    mean = torch.mean(z, dim=0, keepdim=True)
    
    # Center the batch (subtract mean from each element)
    centered_batch = z - mean

    # Compute the covariance (dot product of centered vectors, normalized by batch size)
    cov = torch.mm(centered_batch.T, centered_batch) / (z.size(0) - 1)

    # Compute standard deviations (L2 norm of each column)
    std_dev = torch.sqrt(torch.sum(centered_batch**2, dim=0) / (z.size(0) - 1))

    # Compute correlation matrix
    correlation_matrix = cov / torch.outer(std_dev, std_dev)

    # Loss is 1 - mean absolute correlation (excluding diagonal)
    # correlation_loss = 1 - (torch.sum(torch.abs(correlation_matrix)) - z.size(1)) / (z.size(1)**2 - z.size(1))

    # Create a mask to exclude the diagonal
    mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device).bool()
    
    # Exclude the diagonal and square the remaining elements
    squared_correlation = correlation_matrix[~mask].pow(2)

    # Loss is the mean of the squared correlation elements (excluding the diagonal)
    correlation_loss = squared_correlation.mean()
    
    # import pdb; pdb.set_trace()
    return correlation_loss

