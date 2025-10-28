import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

transform_MNIST = transforms.Compose([
   transforms.RandomRotation(20),  # Rotate the image by up to 20 degrees
   transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Random translations
   transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random color jitter
   transforms.ToTensor(),  # Convert to Tensor
])

output_folder1 = "MNIST_Augmentations_1"
output_folder2 = "MNIST_Augmentations_2"

os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

mnist_dataset = datasets.MNIST(root="./data", train=True, download=True)

def save_image(tensor, path):
   img = transforms.ToPILImage()(tensor)
   img.save(path)

for idx, (image, label) in enumerate(tqdm(mnist_dataset)):
    aug1 = transform_MNIST(image)
    aug2 = transform_MNIST(image)

    base_name = f"image_{idx:05d}"  
    aug1_path = os.path.join(output_folder1, f"{base_name}_aug1_{label}.png")
    aug2_path = os.path.join(output_folder2, f"{base_name}_aug2_{label}.png")

    save_image(aug1, aug1_path)
    save_image(aug2, aug2_path)

    if idx >= 10000:
        break

#    if idx % 100 == 0:
#        print(f"Processed {idx} images...")
    