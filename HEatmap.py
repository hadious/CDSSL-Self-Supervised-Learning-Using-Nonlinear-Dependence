import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function to load the dataset
def load_dataset(dataset_name, batch_size=32):
    if dataset_name == "CIFAR":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        labels = dataset.targets
    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        labels = dataset.targets.numpy()
    else:
        raise ValueError("Dataset not supported. Choose 'CIFAR' or 'MNIST'.")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, labels

# Function to generate embeddings
def get_embeddings(encoder, dataloader):
    embeddings_list = []
    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch  # Only input images are needed
            embeddings = encoder(inputs).cpu().numpy()  # Get embeddings
            embeddings_list.append(embeddings)
    return np.vstack(embeddings_list)

# Function to subsample embeddings and labels
def subsample_embeddings(embeddings, labels, num_samples=500):
    n_samples = embeddings.shape[0]
    if n_samples > num_samples:
        indices = np.random.choice(n_samples, num_samples, replace=False)
        return embeddings[indices], labels[:n_samples][indices]
    return embeddings, labels[:n_samples]

# Function to compute HSIC
def hsic(X, Y, sigma=1.0):
    K_X = rbf_kernel(X, gamma=1.0 / (2 * sigma ** 2))
    K_Y = rbf_kernel(Y, gamma=1.0 / (2 * sigma ** 2))
    n = K_X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H
    hsic_value = np.trace(K_X_centered @ K_Y_centered) / (n - 1) ** 2
    return hsic_value

# Function to compute kernel alignment with ground truth
def ground_truth_kernel(labels):
    n = len(labels)
    K_gt = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_gt[i, j] = 1 if labels[i] == labels[j] else 0
    return K_gt

# Function to compute difference metrics
def compute_difference_metrics(kernel1, kernel2):
    difference = np.abs(kernel1 - kernel2)
    return {
        "mean_difference": np.mean(difference),
        "max_difference": np.max(difference),
        "variance_difference": np.var(difference),
    }

# Function to compute and plot heatmaps
def plot_heatmaps(embeddings1, embeddings2, method_names=("CDSSL", "VICReg")):
    # Compute kernels
    gamma = 1 / embeddings1.shape[1]
    rbf_kernel1 = rbf_kernel(embeddings1, gamma=gamma)
    rbf_kernel2 = rbf_kernel(embeddings2, gamma=gamma)
    cosine_kernel1 = cosine_similarity(embeddings1)
    cosine_kernel2 = cosine_similarity(embeddings2)

    # RBF kernel heatmaps
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    sns.heatmap(rbf_kernel1, cmap="viridis", cbar=True)
    plt.title(f"RBF Kernel - {method_names[0]}")

    plt.subplot(2, 3, 2)
    sns.heatmap(rbf_kernel2, cmap="viridis", cbar=True)
    plt.title(f"RBF Kernel - {method_names[1]}")

    plt.subplot(2, 3, 3)
    sns.heatmap(np.abs(rbf_kernel1 - rbf_kernel2), cmap="inferno", cbar=True)
    plt.title(f"RBF Kernel Difference")

    # Cosine similarity heatmaps
    plt.subplot(2, 3, 4)
    sns.heatmap(cosine_kernel1, cmap="coolwarm", cbar=True)
    plt.title(f"Cosine Similarity - {method_names[0]}")

    plt.subplot(2, 3, 5)
    sns.heatmap(cosine_kernel2, cmap="coolwarm", cbar=True)
    plt.title(f"Cosine Similarity - {method_names[1]}")

    plt.subplot(2, 3, 6)
    sns.heatmap(np.abs(cosine_kernel1 - cosine_kernel2), cmap="inferno", cbar=True)
    plt.title(f"Cosine Similarity Difference")

    plt.tight_layout()
    plt.show()

# Main code
if __name__ == "__main__":
    # User settings
    dataset_name = "MNIST"  # Change to "CIFAR" for CIFAR
    encoder_path1 = "/home/hadi/Desktop/WorkSpace/SSL/LMA-OT/models/Model_MNIST_OURS_100.pth"  # Path to your method
    encoder_path2 = "/home/hadi/Desktop/WorkSpace/SSL/LMA-OT/models/Model_MNIST_VIC_100.pth"  # Path to baseline method

    # Load dataset and labels
    dataloader, labels = load_dataset(dataset_name)

    # Load encoders
    if dataset_name == "CIFAR":
        from utils.models import encoder_expander
        encoder1 = encoder_expander(encoder_dim=512, projector_dim=1024).encoder
    elif dataset_name == "MNIST":
        from utils.models import Encoder
        encoder1 = Encoder(latent_dim=512)
    encoder1.load_state_dict(torch.load(encoder_path1))
    encoder1.eval()

    try:
        if dataset_name == "CIFAR":
            encoder2 = encoder_expander(encoder_dim=512, projector_dim=1024).encoder
        elif dataset_name == "MNIST":
            encoder2 = Encoder(latent_dim=512)
        encoder2.load_state_dict(torch.load(encoder_path2))
        encoder2.eval()
    except FileNotFoundError:
        encoder2 = None

    # Generate embeddings
    embeddings1 = get_embeddings(encoder1, dataloader)
    embeddings2 = get_embeddings(encoder2, dataloader) if encoder2 else None

    # Subsample embeddings and labels
    embeddings1, labels1 = subsample_embeddings(embeddings1, labels, num_samples=25)
    if embeddings2 is not None:
        embeddings2, _ = subsample_embeddings(embeddings2, labels, num_samples=25)

    # Compute ground truth kernel for subsampled labels
    K_gt = ground_truth_kernel(labels1)

    # Compute RBF kernel for your embeddings
    K_yours = rbf_kernel(embeddings1, gamma=1.0 / (2 * 1.0 ** 2))
    alignment_yours = np.linalg.norm(K_yours - K_gt, ord="fro")

    # Compute alignment for baseline if available
    if embeddings2 is not None:
        K_baseline = rbf_kernel(embeddings2, gamma=1.0 / (2 * 1.0 ** 2))
        alignment_baseline = np.linalg.norm(K_baseline - K_gt, ord="fro")
        print("\nKernel Alignment Metrics:")
        print(f"  Alignment Error (CDSSL): {alignment_yours:.4f}")
        print(f"  Alignment Error (VICReg): {alignment_baseline:.4f}")

    # Compute HSIC
    if embeddings2 is not None:
        hsic_yours = hsic(embeddings1, embeddings1)
        hsic_baseline = hsic(embeddings2, embeddings2)
        print("\nHSIC Metrics:")
        print(f"  HSIC (CDSSL): {hsic_yours:.4f}")
        print(f"  HSIC (VICReg): {hsic_baseline:.4f}")

    # Compute RBF and cosine difference metrics
    if embeddings2 is not None:
        rbf_diff_metrics = compute_difference_metrics(
            rbf_kernel(embeddings1, gamma=1 / embeddings1.shape[1]),
            rbf_kernel(embeddings2, gamma=1 / embeddings2.shape[1])
        )
        cosine_diff_metrics = compute_difference_metrics(
            cosine_similarity(embeddings1),
            cosine_similarity(embeddings2)
        )
        print("\nRBF Kernel Difference Metrics:")
        print(f"  Mean Difference: {rbf_diff_metrics['mean_difference']:.4f}")
        print(f"  Max Difference: {rbf_diff_metrics['max_difference']:.4f}")
        print(f"  Variance of Difference: {rbf_diff_metrics['variance_difference']:.4f}")

        print("\nCosine Similarity Difference Metrics:")
        print(f"  Mean Difference: {cosine_diff_metrics['mean_difference']:.4f}")
        print(f"  Max Difference: {cosine_diff_metrics['max_difference']:.4f}")
        print(f"  Variance of Difference: {cosine_diff_metrics['variance_difference']:.4f}")

    # Plot heatmaps for both methods and their differences
    if embeddings2 is not None:
        plot_heatmaps(embeddings1, embeddings2, method_names=("CDSSL", "VICReg"))
