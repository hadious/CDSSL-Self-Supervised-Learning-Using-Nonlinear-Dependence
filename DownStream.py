import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# Define the non-linear classification head (MLP)
class ClassificationHead(nn.Module):
    def __init__(self, encoder_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Load Dataset
def load_dataset(dataset_name, batch_size=32):
    if dataset_name == "CIFAR":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        dataset = datasets.STL10(root='./data', split='train', transform=transform, download=False)
        test_dataset = datasets.STL10(root='./data', split='test', transform=transform, download=False)
        # dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        # test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    else:
        raise ValueError("Dataset not supported. Choose 'CIFAR' or 'MNIST'.")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Main Script
if __name__ == "__main__":
    # Settings
    dataset_name = "CIFAR"  # Change to "CIFAR" for CIFAR
    encoder_path = "/home/hadi/Downloads/Model_CIFAR_OURS.pth"
    encoder_dim = 512 
    num_classes = 10
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    train_loader, test_loader = load_dataset(dataset_name, batch_size)

    # Load Encoder
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
    
    encoder.eval()  # Freeze encoder during training
    encoder.to(device)

    # Create Model with MLP Head
    classification_head = ClassificationHead(encoder_dim=encoder_dim, num_classes=num_classes).to(device)
    model = nn.Sequential(encoder, classification_head)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_head.parameters(), lr=learning_rate)

    # Train and Evaluate
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

    # Final Test Accuracy
    final_accuracy = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
