import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Function to load dataset
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
def get_embeddings(encoder, dataloader, device):
    embeddings_list = []
    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch  # Only input images are needed
            inputs = inputs.to(device)
            embeddings = encoder(inputs).cpu().numpy()
            embeddings_list.append(embeddings)
    return np.vstack(embeddings_list)

# Clustering and Evaluation Function
def clustering_evaluation(embeddings, labels, num_clusters):
    """
    Perform clustering on embeddings and evaluate with NMI and ARI.
    """
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)



    # Compute evaluation metrics
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)

    return nmi, ari

# Main script
if __name__ == "__main__":
    # Settings
    dataset_name = "CIFAR"  # Change to "CIFAR" for CIFAR
    encoder_path = "/home/hadi/Desktop/WorkSpace/SSL/LMA-OT/models/Model_CIFAR_VIC_500.pth"
    encoder_dim = 512
    num_clusters = 10  # Number of ground truth classes
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataloader, labels = load_dataset(dataset_name, batch_size)

    # Load encoder
    if dataset_name == "CIFAR":
        from utils.models import encoder_expander
        encoder_exp = encoder_expander(encoder_dim=512, projector_dim=1024)
        encoder = encoder_exp.encoder
    elif dataset_name == "MNIST":
        from utils.models import Encoder
        encoder = Encoder(latent_dim=512)


    state_dict = torch.load(encoder_path)
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

    encoder.load_state_dict(encoder_state_dict, strict=True)

    # encoder.load_state_dict(torch.load(encoder_path))
    encoder.to(device)

    # Generate embeddings
    embeddings = get_embeddings(encoder, dataloader, device)

    # Perform clustering and evaluate
    nmi, ari = clustering_evaluation(embeddings, labels, num_clusters)
    print(f"Clustering Evaluation Metrics:")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
