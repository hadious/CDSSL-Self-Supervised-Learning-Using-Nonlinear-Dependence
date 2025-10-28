# Import required libraries
from pathlib import Path
import argparse
import json
import os
import sys
import time
from torch import nn, optim
import torch
from torchvision import datasets, transforms
import resnet  # Assuming your custom resnet is here

# 15.18% - 


def get_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on MNIST")


    parser.add_argument("--data-dir", type=Path, default="./data", help="path to dataset")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10, TINYIMAGENET"],default="CIFAR10", help="Specify which dataset to use: MNIST or CIFAR10")
    # Data
    # parser.add_argument("--data-dir", type=Path, default="./data", help="path to dataset")

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, default="exp/resnet34.pth", help="path to pretrained model")
    parser.add_argument("--exp-dir", default="./checkpoint/lincls/", type=Path, help="path to checkpoint directory")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")

    # Model
    parser.add_argument("--arch", type=str, default="resnet34")

    # Optimizer and training settings
    parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--batch-size", default=256, type=int, help="mini-batch size")
    parser.add_argument("--lr-backbone", default=0.0, type=float, help="backbone base learning rate")
    parser.add_argument("--lr-head", default=0.3, type=float, help="classifier base learning rate")
    parser.add_argument("--weight-decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"), help="finetune or freeze resnet weights")

    # Running
    parser.add_argument("--workers", default=8, type=int, help="number of data loader workers")
    
    return parser

def main():
    parser = get_arguments()
    args = parser.parse_args()

    # Single GPU setup
    args.rank = 0
    args.world_size = 1

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    gpu = torch.device('cuda')

    # Load the backbone (ResNet) model and pretrained weights
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    backbone.load_state_dict(state_dict, strict=False)

    if args.dataset == "MNIST":
        num_classes = 10  # MNIST has 10 classes
    elif args.dataset == "CIFAR10":
        num_classes = 10  # CIFAR-10 also has 10 classes
    elif args.dataset == "TINYIMAGENET":
        num_classes = 200
    elif args.dataset == "MINIIMAGENET":
        num_classes = 100 

    # Add the classification head for MNIST (10 classes)
    head = nn.Linear(embedding, num_classes) 
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    model = nn.Sequential(backbone, head).cuda(gpu)

    # Freeze or finetune the backbone
    if args.weights == "freeze":
        backbone.requires_grad_(False)
    else:
        backbone.requires_grad_(True)

    # Set up optimizer and criterion
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    if args.dataset == "MNIST":
        # Load the MNIST dataset
        normalize = transforms.Normalize((0.1307,), (0.3081,))  # Normalization for MNIST
        train_dataset = datasets.MNIST(
            root=args.data_dir, train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )
        val_dataset = datasets.MNIST(
            root=args.data_dir, train=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    elif args.dataset == "CIFAR10":
        from torchvision import transforms
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalization for CIFAR-10
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
                ,normalize,
            ])
        )
        val_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False,
            transform=transforms.Compose([
                transforms.ToTensor()
                , normalize,
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    elif args.dataset == "TINYIMAGENET":
        from torchvision import transforms
        from torch.utils.data import DataLoader
        from Dataset_handler import TinyImageNetDataset

        normalize = transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))

        # Create dataset instances
        train_dataset = TinyImageNetDataset(
            root_dir=args.data_dir / 'tiny-imagenet-200' / 'train',
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )
        val_dataset = TinyImageNetDataset(
            root_dir=args.data_dir / 'tiny-imagenet-200' / 'val',
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
            split='val'
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )
    elif args.dataset == "MINIIMAGENET":

        from torchvision import transforms
        from torch.utils.data import DataLoader
        from Dataset_handler import MiniImageNetDataset

        normalize = transforms.Normalize((0.471, 0.450, 0.403), (0.278, 0.268, 0.284))

        train_dir = './data/Mini_Imagenet/'
        val_dir = './data/Mini_Imagenet/'

        train_dataset = MiniImageNetDataset(root_dir=train_dir,
                                             transform= transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), normalize]),
                                                                            split='train')

        val_dataset = MiniImageNetDataset(root_dir=val_dir, transform= transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), normalize]), split='val')

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
        )

    # Training and evaluation loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train() if args.weights == "finetune" else model.eval()
        
        # Training loop
        for step, (images, target) in enumerate(train_loader):
            images, target = images.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.print_freq == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{step}], Loss: {loss.item():.4f}")

        # Evaluation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.cuda(gpu, non_blocking=True), target.cuda(gpu, non_blocking=True)

                output = model(images)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

        best_acc = max(best_acc, accuracy)
        scheduler.step()

        # Save the best model
        if accuracy > best_acc:
            print(f"New best accuracy: {best_acc:.2f}%")
            torch.save(model.state_dict(), args.exp_dir / "best_model.pth")

if __name__ == "__main__":
    main()
