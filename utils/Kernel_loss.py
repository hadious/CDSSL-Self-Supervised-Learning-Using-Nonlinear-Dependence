import torch

class KernelVICRegLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, eps=1e-4, min_variance=1.0):
        """
        Kernel VICReg loss function.
        
        Args:
        - alpha: Weight for the invariance (MSE) loss.
        - beta: Weight for the variance loss.
        - gamma: Weight for the covariance loss.
        - eps: Small value to avoid numerical issues.
        - min_variance: Minimum variance target for each dimension in the RKHS.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.min_variance = min_variance

    def forward(self, K_11, K_22, K_12):
        """
        Compute the kernelized VICReg loss.
        
        Args:
        - K_11: Kernel matrix for the first batch (n x n).
        - K_22: Kernel matrix for the second batch (n x n).
        - K_12: Cross-kernel matrix between batches (n x n).
        
        Returns:
        - Total kernel VICReg loss.
        """
        # Number of samples (batch size)
        n = K_11.shape[0]
        
        # Centering matrix
        H = torch.eye(n, device=K_11.device) - (1.0 / n) * torch.ones((n, n), device=K_11.device)
        
        # Centered kernel matrices
        K_11_c = H @ K_11 @ H
        K_22_c = H @ K_22 @ H
        
        # Invariance (MSE in RKHS)
        L_inv = torch.trace(K_11 + K_22 - 2 * K_12) / n
        
        # Variance in RKHS (eigenvalues of centered kernel matrix)
        L_var = 0
        for kernel in [K_11_c, K_22_c]:
            eigvals = torch.linalg.eigvalsh(kernel)
            variances = torch.sqrt(torch.clamp(eigvals / n, min=self.eps))
            L_var += torch.mean(torch.clamp(self.min_variance - variances, min=0) ** 2)
        
        # Covariance in RKHS (off-diagonal elements of centered kernel matrix)
        L_cov = 0
        for kernel in [K_11_c, K_22_c]:
            cov_matrix = kernel / n
            L_cov += torch.sum(cov_matrix ** 2) - torch.sum(torch.diag(cov_matrix) ** 2)
  
        
        return self.alpha * L_inv , self.beta * L_var , self.gamma * L_cov