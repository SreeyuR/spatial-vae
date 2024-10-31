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

# Get the dataset
trainset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train = False, download = True, transform = transforms.ToTensor())
xtrain = trainset.data.numpy()
ytrain = trainset.targets.numpy()
x_val_pre = testset.data[:1000].numpy()
y_val = testset.targets[:1000].numpy()

# Creating x_train and y_train with 1000 images from each class and binarizing the pixels
count = np.zeros(10)
idx = []
for i in range(0, len(ytrain)):
  for j in range(10):
    if(ytrain[i] == j):
      count[j] += 1
      if(count[j]<=1000):
        idx = np.append(idx, i)

y_train = ytrain[idx.astype('int')]
x_train_pre = xtrain[idx.astype('int')]

# Keep Original Dimensions
r, h, w = x_train_pre.shape  # Get the original dimensions of x_train_pre
x_train = np.zeros([r, h, w])
for i in range(r):
    x_train[i] = x_train_pre[i].astype('float32')  # Copy without resizing

r, h, w = x_val_pre.shape  # Get the original dimensions of x_val_pre
x_val = np.zeros([r, h, w])
for i in range(r):
    x_val[i] = x_val_pre[i].astype('float32')  # Copy without resizing

# Binarizing
x_train = np.where(x_train > 128, 1, 0)
x_val = np.where(x_val > 128, 1, 0)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

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
print(labels)


# Function to display images from a DataLoader
def show_images(data_loader, num_images=16):
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))  # Get the first batch

    # Set the number of images to display
    num_images = min(num_images, images.size(0))  # Ensure we don't exceed batch size

    # Create a grid of images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  # Adjust grid size as needed
    axes = axes.flatten()

    for i in range(num_images):
        # Get the image and label
        image = images[i].numpy().squeeze()  # Convert to NumPy and remove color channel
        label = labels[i].item()  # Get the label

        # Plot the image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

# Display images from the rotated train loader
show_images(trainloader_rotated)

# Display images from the translated train loader
show_images(trainloader_rotated_translated)

# DIMENSIONS
x_dim = 784 # 28 by 28
hidden_dim = 500
z_dim = 50


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder layers
        self.enc_l1 = nn.Linear(x_dim, hidden_dim)
        self.enc_l2_mu = nn.Linear(hidden_dim, z_dim)  # Mean
        self.enc_l2_log_var = nn.Linear(hidden_dim, z_dim)  # Log variance

        # Decoder layers
        self.dec_l1 = nn.Linear(z_dim + 2, hidden_dim)  # Adjust input size to include spatial coordinates
        self.dec_l2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x):
        x = F.tanh(self.enc_l1(x))
        mu = self.enc_l2_mu(x)
        log_var = self.enc_l2_log_var(x)
        return mu, log_var

    def sample_z_trick_normal(self, mu, log_var):
        std = torch.exp(log_var / 2)
        sample = torch.randn_like(std)
        z = std * sample + mu
        return z

    def decoder(self, z_x_coord):
        out_dec_l1 = F.relu(self.dec_l1(z_x_coord))  # Use concatenated input here
        x_reconstructed = torch.sigmoid(self.dec_l2(out_dec_l1))
        return x_reconstructed

    def forward(self, x, x_coord):
        mu, log_var = self.encoder(x.view(-1, x_dim))  # Flatten the input
        z = self.sample_z_trick_normal(mu, log_var)

        # Concatenate the sampled z and spatial coordinates
        z_x_coord = torch.cat([z, x_coord], dim=1)

        # Pass the concatenated input through the decoder
        x_reconstructed = self.decoder(z_x_coord)
        return x_reconstructed, z, mu, log_var

# Instantiate the model and check if CUDA is available
model = VAE()
if torch.cuda.is_available():
    model.cuda()

import numpy as np

# Step 1: Define the grid size
grid_size = 28

# Step 2: Create a 28x28 grid of Cartesian coordinates
# Create a linear space from -1 to 1, with grid_size points
x = np.linspace(-1, 1, grid_size)  # X-coordinates
y = np.linspace(-1, 1, grid_size)  # Y-coordinates

# Step 3: Create a meshgrid to get all combinations of (x, y)
X, Y = np.meshgrid(x, y)

# Step 4: Stack the coordinates into a tensor (or you can use a list)
coordinates_tensor = np.stack((X, Y), axis=-1)  # Shape (28, 28, 2)
coordinates_list = np.array(coordinates_tensor).reshape(-1, 2)  # Shape (784, 2)
x_coord = coordinates_tensor.reshape(-1, 2)

# Print the coordinates tensor and list
print("Coordinates Tensor:\n", coordinates_tensor)
print("\nCoordinates List:\n", coordinates_list)


# Instantiate the model and check if CUDA is available
model = VAE()
if torch.cuda.is_available():
    model.cuda()

# Define batch size
batch_size = 64  # Example batch size

# Example input data (randomly generated for demonstration)
x_input = torch.randn(batch_size, x_dim)  # Replace this with your actual input tensor

# Move input to GPU if available
if torch.cuda.is_available():
    x_input = x_input.cuda()

# Convert x_coord to a PyTorch tensor
x_coord_tensor = torch.tensor(x_coord, dtype=torch.float32)

# Expand x_coord_tensor to match the batch size and reshape it
x_coord_expanded = x_coord_tensor.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, -1)

# Move expanded coordinates to GPU if available
if torch.cuda.is_available():
    x_coord_expanded = x_coord_expanded.cuda()

# Forward pass through the model
x_reconstructed, z, mu, log_var = model(x_input, x_coord_expanded)

# You can print or further process the outputs here
print("Reconstructed Output Shape:", x_reconstructed.shape)
print("Latent Variable Shape:", z.shape)
print("Mean Shape:", mu.shape)
print("Log Variance Shape:", log_var.shape)
