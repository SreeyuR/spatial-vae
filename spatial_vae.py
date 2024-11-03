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

from data import get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

trainloader_rotated, trainloader_rotated_translated, testloader_rotated, testloader_rotated_translated = get_data()


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

        # Set dimensions
        x_dim = 784
        hidden_dim = 256
        z_dim = 50

        # Encoder
        self.enc_l1 = nn.Linear(x_dim, hidden_dim)
        self.enc_l2_mu = nn.Linear(hidden_dim, z_dim)  # mu
        self.enc_l2_log_var = nn.Linear(hidden_dim, z_dim)  # log variance

        # Decoder
        self.dec_l1 = nn.Linear(z_dim, hidden_dim)
        self.dec_l2 = nn.Linear(hidden_dim, x_dim)

    def encoder(self, x):
        x = torch.relu(self.enc_l1(x))
        mu = self.enc_l2_mu(x)
        log_var = self.enc_l2_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Equivalent to torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def decoder(self, z, x_coords):

        # Concatenate z & x_coords together
        z = torch.cat((z, x_coords), dim=-1)

        x = torch.relu(self.dec_l1(z))
        x = torch.sigmoid(self.dec_l2(x))  # Sigmoid to match MNIST pixel values

        return x

    def spatial_rotate_x(self, x_coords, theta):
        b = theta.shape[0]  # batch size

        # Initialize rotation_matrix with float type
        rotation_matrix = torch.zeros((b, 2, 2), dtype=torch.float32)

        rotation_matrix[:, 0, 0] = torch.cos(theta).squeeze()
        rotation_matrix[:, 0, 1] = torch.sin(theta).squeeze()
        rotation_matrix[:, 1, 0] = -torch.sin(theta).squeeze()
        rotation_matrix[:, 1, 1] = torch.cos(theta).squeeze()

        # Perform batch matrix multiplication
        rotated_x_coords = torch.matmul(x_coords, rotation_matrix)

        return rotated_x_coords

    def translate_x(self, x_coords, delta_x):
        return x_coords + delta_x  # TO DO - CONSIDER delta_x scaling factor

    def forward(self, x, x_coords, rotate=False, translate=False):
        mu, log_var = self.encoder(x.view(-1, x_dim))
        theta_prior_std = torch.pi  # DEFINITELY NOT THE RIGHT DIMENSION - HELP PLEASE
        theta_mu = mu[:, 0]
        theta_logvar = log_var[:, 0]

        # mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        if rotate:
            x_coords = self.spatial_rotate_x(x_coords, z[:, 0])  # z[0] represents latent theta

        if translate:
            x_coords = self.translate_x(x_coords, z[:, 1:3])  # z[1] and z[2] represent (x_0, x_1)

        y = self.decoder(z, x_coords)

        # PARAMS NEEDED
        return y, z, mu, logvar, theta_prior_std, theta_mu, theta_logvar  # figure out how to get all these params

def get_grid(grid_size_hor=28, grid_size_vert=28):
    # 28x28 grid of Cartesian coordinates: linear space from -1 to 1, with grid_size points
    x0 = np.linspace(-1, 1, grid_size_hor)  # X-coordinates
    x1 = np.linspace(-1, 1, grid_size_vert)  # Y-coordinates
    # meshgrid to get all combinations of (x, y)
    X0, X1 = np.meshgrid(x0, x1)
    # stack x0 and x1 against each other for each col
    coordinates_tensor = np.stack((X0.ravel(), X1.ravel()), axis=1)  # (grid_size_hor * grid_size_vert, 2)
    return coordinates_tensor

batch_size = 64

# Define the loss function - MODIFIED
def loss_function(output, x, mu, logvar, theta_mu, theta_prior, theta_logvar, rotate, translate_x, train_loss, alpha_recon=1, beta_kl=0.002):
    recon_loss = F.mse_loss(output, x.view(-1, x_dim), reduction='sum') #reconstruction term
    #recon_loss = F.binary_cross_entropy(output, x.view(-1, x_dim), reduction='sum')
    z_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # delta_x_kl_loss represented in this
    kl_loss = z_kl_loss
    if rotate:
      theta_kl_loss = -0.5 + -theta_logvar + np.log(theta_prior) + (theta_logvar.exp() + theta_mu.pow(2))/(2 * theta_prior.pow(2)) # question about u^2
      kl_loss = z_kl_loss + theta_kl_loss

    # For graphing purposes
    if train_loss:
      reconstruction_losses.append(recon_loss.item()/batch_size)
      kl_losses.append(kl_loss.item()/batch_size)

    return (alpha_recon * recon_loss + beta_kl * kl_loss)/batch_size


model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
reconstruction_losses = []
kl_losses = []
weight_updates = 0
num_epochs= 100
losses = []
epochs = []
val_losses = []

# Train the model
for epoch in range(num_epochs):
    epochs.append(epoch)
    epoch_loss = 0
    for batch in trainloader_rotated:

        # OUR CURRENT UNDERSTANDING
        # (B, P_b, stuff)
        # B --> number of images in a batch
        # P_b --> number of pixels in a batch
        # stuff --> ? (theta, delta_x, z)

        # Zero the gradients
        optimizer.zero_grad()

        # Get batch
        x = batch[0]

        # Forward pass - THIS NEEDS TO BE MODIFIED
        output, z, mu, logvar, theta_mu, theta_prior, theta_logvar, theta_mu = model(x)

        # Calculate loss - THIS NEEDS TO BE MODIFIED
        loss = loss_function(output, x, mu, logvar, theta_mu, theta_prior, theta_logvar, theta_mu, True)


        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Add batch loss to epoch loss
        epoch_loss += loss.item()
        weight_updates += 1

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to compute gradients during validation
        for batch in testloader_rotated:
            x = batch[0]

            # Forward pass
            output, z, mu, logvar = model(x)

            # Calculate validation loss
            loss = loss_function(output, x, mu, logvar, False)
            val_loss += loss.item()

    # Average validation loss for the epoch
    avg_val_loss = val_loss / len(testloader_rotated)

    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(x)}, Validation Loss: {avg_val_loss}")
    losses.append(epoch_loss/len(x))
    val_losses.append(avg_val_loss)
