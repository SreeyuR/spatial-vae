# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import cv2
import matplotlib.pyplot as plt
import math
import torch.optim as optim
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor())
# testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transforms.ToTensor())


def flatten_pixels(dataset, grid_size_hor=28, grid_size_vert=28):
    dataset.data = dataset.data.view(-1, grid_size_hor*grid_size_vert) # num_images, flattened_pixels

class RandomRotationTranslation:
    def __init__(self, rotation_std=np.pi**2 / 16, translation_std_small=1.42, translation_std_large=142, variant="rotated"):
        self.rotation_std = np.sqrt(rotation_std)
        self.translation_std = translation_std_small if variant == "rotated" else translation_std_large
        self.variant = variant

    def __call__(self, img):
        # Sample a rotation angle from N(0, π^2 / 16)
        angle = np.random.normal(0, self.rotation_std) * (180 / np.pi)  # Convert radians to degrees

        # Sample translations from N(0, translation_std)
        translation_x = np.random.normal(0, self.translation_std)
        translation_y = np.random.normal(0, self.translation_std)

        # Apply transformations
        img = transforms.functional.rotate(img, angle)  # Rotate the image
        img = transforms.functional.affine(img, angle=0, translate=(translation_x, translation_y), scale=1, shear=0)  # Apply translation

        return img

# Define a custom dataset
class TransformedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, variant="rotated"):
        super().__init__(root, train=train, transform=transform, download=download)
        self.variant = variant
        self.custom_transform = RandomRotationTranslation(variant=self.variant)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if not isinstance(img, Image.Image):
          img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        img = self.custom_transform(img)
        img = (img > 0.5).float()
        img = img.view(-1)
        return img, target

# Set up transformations and load datasets
train_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    # Normalize pixel values (optional, depending on model requirements)
    #transforms.Normalize((0.5,), (0.5,))
])

# Load the datasets with transformations
trainset_rotated = TransformedMNIST(root='./data', train=True, download=True, transform=train_transform, variant="rotated")
trainset_rotated_translated = TransformedMNIST(root='./data', train=True, download=True, transform=train_transform, variant="rotated_translated")

testset_rotated = TransformedMNIST(root='./data', train=False, download=True, transform=train_transform, variant="rotated")
testset_rotated_translated = TransformedMNIST(root='./data', train=False, download=True, transform=train_transform, variant="rotated_translated")

########################################

flatten_pixels(trainset_rotated)
flatten_pixels(trainset_rotated_translated)
flatten_pixels(testset_rotated)
flatten_pixels(testset_rotated_translated)


print(f'flattened: {trainset_rotated.data.shape}')

#************************************************************************
targets = trainset_rotated.targets
class_counts = torch.zeros(10, dtype=torch.int64)
indices = []

for i, target in enumerate(targets):
    label = target.item()
    if class_counts[label] < 1000:
        indices.append(i)
        class_counts[label] += 1
    if class_counts.sum() >= 10000:
        break

# Create a subset dataset using the filtered indices
trainset_rotated = Subset(trainset_rotated, indices)

#*************************************************************************
targets = testset_rotated.targets
class_counts = torch.zeros(10, dtype=torch.int64)
indices = []

for i, target in enumerate(targets):
    label = target.item()
    if class_counts[label] < 1000:
        indices.append(i)
        class_counts[label] += 1
    if class_counts.sum() >= 10000:
        break

# Create a subset dataset using the filtered indices
testset_rotated = Subset(testset_rotated, indices)

#*********************************************************************
targets = trainset_rotated_translated.targets
class_counts = torch.zeros(10, dtype=torch.int64)
indices = []

for i, target in enumerate(targets):
    label = target.item()
    if class_counts[label] < 1000:
        indices.append(i)
        class_counts[label] += 1
    if class_counts.sum() >= 10000:
        break

# Create a subset dataset using the filtered indices
trainset_rotated_translated = Subset(trainset_rotated_translated, indices)
#**************************************************************************

targets = testset_rotated_translated.targets
class_counts = torch.zeros(10, dtype=torch.int64)
indices = []

for i, target in enumerate(targets):
    label = target.item()
    if class_counts[label] < 1000:
        indices.append(i)
        class_counts[label] += 1
    if class_counts.sum() >= 10000:
        break

# Create a subset dataset using the filtered indices
testset_rotated_translated = Subset(testset_rotated_translated, indices)
#***********************************************************************



# Define DataLoaders
trainloader_rotated = torch.utils.data.DataLoader(trainset_rotated, batch_size=64, shuffle=True)
trainloader_rotated_translated = torch.utils.data.DataLoader(trainset_rotated_translated, batch_size=64, shuffle=True)

testloader_rotated = torch.utils.data.DataLoader(testset_rotated, batch_size=64, shuffle=True)
testloader_rotated_translated = torch.utils.data.DataLoader(testset_rotated_translated, batch_size=64, shuffle=True)

# Test loading one batch to check transformations
data_iter = iter(trainloader_rotated)
images, labels = next(data_iter)
print(images.shape)  # Should be (64, 1, 28, 28) for MNIST images
# print(labels)

def get_data():
    return trainloader_rotated, trainloader_rotated_translated, testloader_rotated, testloader_rotated_translated