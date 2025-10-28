import torch 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import os
from datetime import datetime
from torch.optim import Optimizer
import umap
from typing import Optional, List
import sys
from contextlib import contextmanager


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Resuming from epoch {epoch + 1} with loss {loss}")
        return epoch + 1  # Start from the next epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0  # Start from epoch 0


def save_checkpoint(epoch, model, optimizer, loss, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def plot_embedding_length(embedding_lengths: List[float], save_path: None):
    n_epochs = len(embedding_lengths)
    plt.figure(figsize=(8, 6))
    plt.plot(range(n_epochs), embedding_lengths, marker='o')
    plt.title('Average Embedding Length per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Embedding Length')
    plt.grid(True)
    if save_path is None:
        plt.show()
    else:
        print(f"Saving the plot of loss to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

def plot_loss_values(loss_epochs: List[float], save_path: None):
    n_epochs = len(loss_epochs)
    plt.figure(figsize=(8, 6))
    # import pdb; pdb.set_trace()
    plt.plot(range(n_epochs), loss_epochs, marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_path is None:
        plt.show()
    else:
        print(f"Saving the plot of loss to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

def visualize_latent_space_with_pca_and_tsne(encoder, data_loader, device, save_path = None, n_batches_to_use: Optional[int] = None):
    encoder.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            # Get latent representations
            latent_representations = encoder.encoder(images)

            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    if n_batches_to_use is not None:
        latent_vectors = latent_vectors[:n_batches_to_use]
        labels_list = labels_list[:n_batches_to_use]

    # Convert to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # # PCA visualization
    # pca = PCA(n_components=2)
    # pca_results = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(16, 8))

    # # Plot PCA results
    # plt.subplot(1, 2, 1)
    # scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    # plt.legend(*scatter.legend_elements(), title="Classes")
    # plt.title("PCA Visualization of Latent Space")

    # t-SNE visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    tsne_results = tsne.fit_transform(latent_vectors)

    # UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(latent_vectors)

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Latent Space")

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Visualization of Latent Space")

    if save_path is None:
        plt.show()
    else:
        print(f"Saving the plot of embedding to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

def visualize_latent_space_with_tsne(encoder, data_loader, device):
    encoder.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            # Get latent representations
            latent_representations = encoder(images)

            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    # Convert to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # t-SNE visualization
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    tsne_results = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Latent Space")

    plt.show()


def visualize_latent_space_with_umap(encoder, data_loader, device):
    encoder.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            # Get latent representations
            latent_representations = encoder(images)

            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    # Convert to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # UMAP visualization
    umap_reducer = umap.UMAP(n_components=2, random_state=0)
    umap_results = umap_reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Visualization of Latent Space")

    plt.show()

# Function to evaluate learned representations using a k-NN classifier
def evaluate_knn(encoder, train_loader, test_loader, k=5, device='cpu'):
    encoder.eval()
    train_embeddings, train_labels = [], []
    test_embeddings, test_labels = [], []

    # Collect train set embeddings
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = encoder(images).cpu().numpy()
            train_embeddings.append(embeddings)
            train_labels.append(labels.cpu().numpy())

    # Collect test set embeddings
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = encoder(images).cpu().numpy()
            test_embeddings.append(embeddings)
            test_labels.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Train k-NN classifier on train embeddings
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_embeddings, train_labels)

    # Predict on test set
    test_preds = knn.predict(test_embeddings)

    # Compute k-NN accuracy
    knn_accuracy = accuracy_score(test_labels, test_preds)
    print(f"k-NN Classification Accuracy (k={k}): {knn_accuracy:.4f}")

    # Additional clustering metrics
    ari = adjusted_rand_score(test_labels, test_preds)
    nmi = normalized_mutual_info_score(test_labels, test_preds)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

    return knn_accuracy, ari, nmi



class LARS(Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta,
            weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter
        )
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def show_sample_images(data_loader, num_images=8):
    """
    Displays a grid of images and their corresponding numerical labels from a DataLoader.

    Args:
        data_loader (DataLoader): The DataLoader to sample images from.
        num_images (int): Number of images to display.
    """
    # Get a batch of images and labels from the DataLoader
    images, labels = next(iter(data_loader))

    # Limit to the specified number of images
    images = images[:num_images]
    labels = labels[:num_images]

    # Plot the images in a grid
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        image = images[i].permute(1, 2, 0)  # Change (C, H, W) to (H, W, C) for Matplotlib
        plt.imshow(image)
        plt.title(f"Label: {labels[i].item()}")  # Display the numerical label
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_model(model, directory="models", prefix="encoder"):
    os.makedirs(directory, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pth"
    file_path = os.path.join(directory, filename)
    
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, filepath):
    # Load the model state_dict
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")





def visualize_latent_space_umap(encoder, data_loader, device, save_path=None, n_batches_to_use: Optional[int] = None):
    encoder.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            # Get latent representations
            try:
                latent_representations = encoder.encoder(images)
            except:
                latent_representations = encoder(images)



            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    if n_batches_to_use is not None:
        latent_vectors = latent_vectors[:n_batches_to_use]
        labels_list = labels_list[:n_batches_to_use]

    # Convert to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # UMAP visualization
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Visualization of Latent Space")

    if save_path is None:
        plt.show()
    else:
        print(f"Saving the plot of embedding to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


@contextmanager
def redirect_terminal_output(file_path):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(file_path, 'w') as file:
            sys.stdout = file
            sys.stderr = file
            yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def visualize_views(view1, view2, num_images=5):
 
    view1 = view1.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (N, H, W, C) format for plotting
    view2 = view2.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (N, H, W, C) format for plotting

    plt.figure(figsize=(12, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow((view1[i] * 0.5) + 0.5)  # Undo normalization for display
        plt.axis('off')
        plt.title("Original View")

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow((view2[i] * 0.5) + 0.5)  # Undo normalization for display
        plt.axis('off')
        plt.title("Augmented View")

    plt.suptitle("Visualization of Original and Augmented Views")
    plt.show()


def smooth_weights(t, T, beta=5):
        w1 = 1 / (1 + np.exp(beta * (t - T / 3)))
        w2 = (1 / (1 + np.exp(-beta * (t - T / 3)))) * (1 / (1 + np.exp(beta * (t - 2 * T / 3))))
        w3 = 1 / (1 + np.exp(-beta * (t - 2 * T / 3)))
        total = w1 + w2 + w3
        return [w1 / total, w2 / total, w3 / total]