import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []
        
        if split == 'train':
            # Training data is organized in class-specific folders
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_dir = os.path.join(train_dir, cls_name, "images")
                for img_name in os.listdir(cls_dir):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        elif split == 'val':
            # Validation data is in a single folder with a label file
            val_dir = os.path.join(root_dir, 'val')
            val_images_dir = os.path.join(val_dir, "images")
            val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
            self.class_to_idx = self._load_val_classes(val_annotations_file)
            self._load_val_labels(val_annotations_file)

            for img_name in os.listdir(val_images_dir):
                img_path = os.path.join(val_images_dir, img_name)
                self.image_paths.append(img_path)
                img_base_name = os.path.basename(img_name)
                self.labels.append(self.class_to_idx.get(img_base_name, -1))  # -1 for missing labels

    def _load_val_classes(self, annotations_file):
        """Creates a dictionary to map class names to indices."""
        class_to_idx = {}
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.split()
                class_name = parts[1]
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(class_to_idx)
        return class_to_idx

    def _load_val_labels(self, annotations_file):
        """Creates a dictionary to map validation image names to class indices."""
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.split()
                img_name = parts[0]
                class_name = parts[1]
                if class_name in self.class_to_idx:
                    self.class_to_idx[img_name] = self.class_to_idx[class_name]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []

        # Load class labels from Labels.json
        labels_file = os.path.join(root_dir, 'Labels.json')
        with open(labels_file, 'r') as f:
            self.class_to_name = json.load(f)

        # Map class IDs to numerical indices
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_to_name.keys())}

        if split == 'train':
            # Load training images from class-specific folders
            train_dir = os.path.join(root_dir, 'train')
            self.classes = sorted(os.listdir(train_dir))
            for cls_name in self.classes:
                cls_dir = os.path.join(train_dir, cls_name)
                if os.path.isdir(cls_dir):  # Check if it's a directory
                    for img_name in os.listdir(cls_dir):
                        img_path = os.path.join(cls_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

        elif split == 'val':
            # Load validation images from subdirectories in the val folder
            val_dir = os.path.join(root_dir, 'val')
            for cls_name in os.listdir(val_dir):
                cls_dir = os.path.join(val_dir, cls_name)
                if os.path.isdir(cls_dir):  # Ensure it's a directory
                    for img_name in os.listdir(cls_dir):
                        img_path = os.path.join(cls_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
