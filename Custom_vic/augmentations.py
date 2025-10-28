# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


# aug.py

import torchvision.transforms as transforms

class MNISTTrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Randomly crop within 80% to 100% of the image size
            transforms.RandomRotation(10),  # Random rotation between -10 and 10 degrees
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean and std
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2  # Return two augmented versions of the same sample




class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2



class CIFAR10TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Randomly crop within 80% to 100% of the image size
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # Color jitter with 80% probability
            transforms.RandomGrayscale(p=0.2),  # Convert to grayscale with 20% probability
            GaussianBlur(p=0.1),  # Apply Gaussian blur with 10% probability
            Solarization(p=0.2),  # Apply solarization with 20% probability
            transforms.ToTensor(),  # Convert to tensor
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10 normalization
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),  # Apply Gaussian blur with 100% probability (more aggressive)
            Solarization(p=0.0),  # Apply solarization with 0% probability (disabled for prime transform)
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2  # Return two augmented versions of the same sample
 

class TinyImageNetTrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]
            )  # Tiny ImageNet normalization
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),  # More aggressive Gaussian blur
            Solarization(p=0.0),  # Disable solarization for the prime transform
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.480, 0.448, 0.398], std=[0.277, 0.269, 0.282]
            )
        ])

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2  # Return two augmented versions of the same sample

class MiniImageNetTrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),  # Resized crop
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),  # Color jitter
            transforms.RandomGrayscale(p=0.2),  # Grayscale with 20% probability
            GaussianBlur(p=0.1),  # Gaussian blur
            Solarization(p=0.2),  # Solarization
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.471, 0.450, 0.403], std=[0.278, 0.268, 0.284]
            )  # Mini-ImageNet normalization
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),  # Aggressive Gaussian blur
            Solarization(p=0.0),  # Disable solarization for prime transform
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.471, 0.450, 0.403], std=[0.278, 0.268, 0.284]
            )
        ])

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2  # Return two augmented versions of the same sample
