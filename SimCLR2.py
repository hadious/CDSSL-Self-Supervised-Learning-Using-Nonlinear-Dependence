import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision.models import resnet18
from tqdm import tqdm
import argparse

class SimCLRTransform:
    def __init__(self, num_channels=3, is_classification=False):
        self.is_classification = is_classification
        self.base_transform = transforms.Compose([
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
            return self.base_transform(x)  # Single transformed image for classification
        else:
            return self.base_transform(x), self.base_transform(x)  # Two views for SimCLR

# ---- Dataset Selection ----
def get_dataset(name, is_classification=False):
    num_channels = 1 if name == 'mnist' else 3
    transform = SimCLRTransform(num_channels, is_classification)

    if name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    elif name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    elif name == 'STL':
        dataset = datasets.STL10(root='./data', split='train', transform=transform, download=False)
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', or 'mnist'.")
    
    return dataset


# ---- Save SimCLR Encoder ----
def save_encoder(model, path="simclr_encoder.pth"):
    torch.save(model.encoder.state_dict(), path)
    print(f"SimCLR encoder saved to {path}")

# ---- Load SimCLR Encoder ----
def load_encoder(model, path="simclr_encoder.pth"):
    model.encoder.load_state_dict(torch.load(path))
    model.encoder.eval()  # Ensure the encoder is in evaluation mode
    print(f"SimCLR encoder loaded from {path}")

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



def pretrain_simclr(model, dataloader, optimizer, epochs=10, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_views, _ = batch  # Correct unpacking
            
            # Ensure we get two views per batch for SimCLR
            x_i, x_j = x_views  # This assumes x_views is a tuple (view1, view2)
            
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

# ---- Evaluation Function (Top-1 and Top-5 Accuracy) ----
def evaluate_classifier(model, dataloader, device='cuda'):
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, top1_preds = torch.max(logits, dim=1)
            _, top5_preds = torch.topk(logits, 5, dim=1)

            top1_correct += (top1_preds == y).sum().item()
            top5_correct += sum([y[i] in top5_preds[i] for i in range(y.size(0))])
            total += y.size(0)

    print(f"Top-1 Accuracy: {top1_correct / total:.4f}, Top-5 Accuracy: {top5_correct / total:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'mnist'], help='Dataset to use')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ---- Load dataset for SimCLR Pretraining ----
    train_dataset = get_dataset(args.dataset, is_classification=False)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    
    input_channels = 1 if args.dataset == 'mnist' else 3
    simclr_model = SimCLR(input_channels=input_channels).to(device)
    optimizer = optim.Adam(simclr_model.parameters(), lr=1e-3, weight_decay=1e-6)

    print(f"Pretraining SimCLR on {args.dataset}...")
    pretrain_simclr(simclr_model, train_loader, optimizer, epochs=100, device=device)
    # load_encoder(simclr_model,'100_simclr_encoder.pth')
    # ---- Load dataset for classification ----
    full_dataset = get_dataset(args.dataset, is_classification=True)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    classifier = LinearClassifier(simclr_model.encoder, num_classes=100).to(device)
    optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    print("Training Linear Classifier...")
    train_classifier(classifier, train_loader, optimizer, criterion, epochs=20, device=device)

    print("Evaluating Classifier...")
    evaluate_classifier(classifier, test_loader, device=device)

if __name__ == "__main__":
    main()
