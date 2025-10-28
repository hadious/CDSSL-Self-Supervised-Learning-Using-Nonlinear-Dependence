import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import resnet18
from tqdm import tqdm
import argparse




def save_encoder(model, path="simclr_encoder.pth"):
    torch.save(model.encoder.state_dict(), path)
    print(f"SimCLR encoder saved to {path}")

# ---- Load SimCLR Encoder ----
def load_encoder(model, path="simclr_encoder.pth"):
    model.encoder.load_state_dict(torch.load(path))
    model.encoder.eval()  # Ensure the encoder is in evaluation mode
    print(f"SimCLR encoder loaded from {path}")



class SimCLRTransform:
    def __init__(self, num_channels=3, is_classification=False):
        self.is_classification = is_classification
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=num_channels) if num_channels == 3 else nn.Identity(),
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __call__(self, x):
        if self.is_classification:
            return self.transform(x)  # Single view for classifier training
        return self.transform(x), self.transform(x)  # Two views for SimCLR

def get_dataset(name, is_classification=False):
    num_channels = 1 if name == 'mnist' else 3
    transform = SimCLRTransform(num_channels, is_classification)
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', or 'mnist'.")
    return dataset

def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)  # Concatenate both views
    similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    
    logits = similarity_matrix / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    
    loss = -torch.sum(labels * log_prob) / (2 * batch_size)
    return loss

# ---- SimCLR Model ----
class SimCLR(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=3):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(pretrained=False)
        
        # Modify first conv layer if input channels != 3
        if input_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder.fc = nn.Identity()  # Remove classification head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

# ---- Linear Classifier ----
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.encoder.eval()  # Freeze encoder weights
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)

def pretrain_simclr(model, dataloader, optimizer, epochs=100, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_i, x_j), _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    # Save encoder after training
    save_encoder(model)


def train_classifier(model, dataloader, optimizer, criterion, epochs=20, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct, total = 0, 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}, Acc: {correct / total:.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist'], help='Dataset to use')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = get_dataset(args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=16)
    
    input_channels = 1 if args.dataset == 'mnist' else 3
    simclr_model = SimCLR(input_channels=input_channels).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    print(f"Pretraining SimCLR on {args.dataset}...")
    pretrain_simclr(simclr_model, train_loader, optimizer, epochs=5, device=device)

    # # Load encoder for classifier training
    # classifier = LinearClassifier(simclr_model.encoder).to(device)
    # load_encoder(classifier)  # Load the pretrained encoder

    # classifier_dataset = get_dataset(args.dataset, is_classification=True)
    # classifier_loader = DataLoader(classifier_dataset, batch_size=512, shuffle=True, num_workers=16)
    # optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    # criterion = nn.CrossEntropyLoss()
    
    # print("Training Linear Classifier...")
    # train_classifier(classifier, classifier_loader, optimizer, criterion, epochs=20, device=device)





def visualize_latent_space_umap(encoder, data_loader, device, save_path=None, n_batches_to_use=100):
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os,umap
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mnist', choices=['cifar10', 'cifar100', 'mnist'], help='Dataset to use')
    # args = parser.parse_args()
    # model = SimCLR(input_channels=3)
    # load_encoder(model,'mnist_simclr_encoder.pth')
    # device =   'cpu'
    # train_dataset = get_dataset(args.dataset)
    # visualize_latent_space_umap(model,train_dataset,device,'mnist_simclr',100)

    main()



#############################
 

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import umap
from torch.utils.data import DataLoader
from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, base_encoder=resnet18, projection_dim=128, input_channels=3):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder(pretrained=False)
        
        # Modify first conv layer if input channels != 3
        if input_channels == 1:
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder.fc = nn.Identity()  # Remove classification head
        
    def forward(self, x):
        h = self.encoder(x)  # Extract 512D representation
        return h


def load_encoder(model, path="simclr_encoder.pth"):
    model.encoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.encoder.eval()
    print(f"SimCLR encoder loaded from {path}")


def get_dataset(name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10' or 'mnist'.")
    
    return dataset


def visualize_latent_space_umap(encoder, data_loader, device, save_path=None, n_batches_to_use=None):
    encoder.eval()
    latent_vectors = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            latent_representations = encoder(images)
            
            latent_vectors.append(latent_representations.cpu().numpy())
            labels_list.append(labels.numpy())

    # Convert to numpy arrays
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_results = reducer.fit_transform(latent_vectors)

    # Plot the embeddings
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels_list, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Visualization of SimCLR Embeddings")

    if save_path is None:
        plt.show()
    else:
        print(f"Saving the plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


if __name__ == "__main__":
    dataset_name = "mnist"  # Change to 'cifar10' if needed
    device = torch.device("cpu")

    # Load dataset and create DataLoader
    dataset = get_dataset(dataset_name)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    # Load pretrained SimCLR model
    model = SimCLR(input_channels=1 if dataset_name == "mnist" else 3)
    load_encoder(model, "simclr_encoder.pth")  # Change path accordingly
    model.to(device)

    # Visualize UMAP
    visualize_latent_space_umap(model, data_loader, device, save_path="./mnist_simclr.png")
