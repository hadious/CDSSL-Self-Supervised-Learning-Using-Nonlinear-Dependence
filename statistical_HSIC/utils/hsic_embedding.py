import numpy as np
import scipy
from sklearn.metrics.pairwise import pairwise_kernels

class HSIC_Embedding:
    def __init__(self, n_components: int | None = None, kernel_on_labels: str = 'rbf'):
        self.n_components = n_components
        self.U = None
        self.kernel_on_labels = kernel_on_labels

    def fit_transform(self, X1: np.ndarray, X2: np.ndarray):
        self.fit(X1=X1, X2=X2)
        X1_transformed = self.transform(X=X1)
        X2_transformed = self.transform(X=X2)
        return X1_transformed, X2_transformed

    def fit(self, X1: np.ndarray, X2: np.ndarray):
        X1 = X1.T
        X2 = X2.T
        d, n = X1.shape
        K_X2 = pairwise_kernels(X=X2.T, Y=X2.T, metric=self.kernel_on_labels)  # kernel on samples of X2
        K_X2_T = pairwise_kernels(X=X2, Y=X2, metric=self.kernel_on_labels)  # kernel on features of X2
        # if self.kernel_on_labels == 'rbf':
        #     K_X2 = pairwise_kernels(X=X2.T, Y=X2.T, metric=self.kernel_on_labels)
        #     K_X2_T = pairwise_kernels(X=X2, Y=X2, metric=self.kernel_on_labels)
        #     # K_X2 = rbf_kernel(X=X2.T)  # kernel on samples of X2
        #     # K_X2_T = rbf_kernel(X=X2)  # kernel on features of X2
        # else:
        #     raise NotImplementedError
        H_n = np.eye(n) - ((1/n) * np.ones((n,n)))
        H_d = np.eye(d) - ((1/d) * np.ones((d,d)))
        S_W = np.zeros((d, d))
        for sample_index in range(n):
            x1 = X1[:, sample_index].reshape((-1, 1))
            x2 = X2[:, sample_index].reshape((-1, 1))
            difference = x1 - x2
            S_W = S_W + (difference @ difference.T)
        S_W = (1/(n-1)) * S_W
        S_W = S_W * 0.01
        # import pdb; pdb.set_trace()
        # M = (1.0 * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (H_d @ K_X2_T @ H_d @ X1 @ X1.T))
        # M = (1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T)) - (0.01 * S_W)
        # epsilon = 0.00001  #--> to prevent singularity of matrix N
        # epsilon = 1e-10
        epsilon = 1e-5 * np.trace(S_W)
        S_W += np.eye(S_W.shape[0]) * epsilon
        # M = (0.0001 * np.linalg.inv(S_W + epsilon*np.eye(S_W.shape[0])) / 10**6) @ ((1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T)))
        # M = (10**-6 * np.linalg.inv(S_W + epsilon*np.eye(S_W.shape[0]))) @ ((1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T)))
        # M = ((1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T)))
        # eig_val, eig_vec = np.linalg.eigh(M)
        # eig_val, eig_vec = scipy.linalg.eigh(((1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T))), S_W)
        eig_val, eig_vec = scipy.linalg.eigh(((1.0 * (1 / (n - 1)**2) * (X1 @ H_n @ K_X2 @ H_n @ X1.T)) - (0.01 * (1 / (d - 1)**2) * (H_d @ K_X2_T @ H_d @ X1 @ X1.T))), np.eye((d)))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def transform(self, X):
        # import pdb; pdb.set_trace()
        X = X.T
        X_transformed = (self.U.T).dot(X)
        X_transformed = X_transformed.T
        return X_transformed