import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Gaussian Encoder
class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout_prob):
        super(GaussianEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        h = F.tanh(self.fc1(x))
        h = self.dropout(h)  # Apply dropout
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        stddev = torch.exp(0.5 * logvar)
        z = mean + stddev * torch.randn_like(stddev)
        return z, mean, logvar

# Define the Bernoulli Decoder
class BernoulliDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim, dropout_prob):
        super(BernoulliDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, z):
        h = F.tanh(self.fc1(z))
        h = self.dropout(h)  # Apply dropout
        y = torch.sigmoid(self.fc2(h))
        return y

# Define the Autoencoder with Gaussian Encoder and Bernoulli Decoder
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout_prob):
        super(VAE, self).__init__()
        self.encoder = GaussianEncoder(input_dim, hidden_dim, z_dim, dropout_prob)
        self.decoder = BernoulliDecoder(z_dim, hidden_dim, input_dim, dropout_prob)
    
    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        y = self.decoder(z)
        return y, mean, logvar

def save_image_grid(images, grid_shape, image_size, output_file):
    """
    Combines a list of images into a grid and saves the result as a single image.

    Args:
    - images: List of images to be placed in a grid (numpy arrays).
    - grid_shape: Tuple (rows, cols) representing the grid dimensions.
    - image_size: Size (width, height) of each image in the grid.
    - output_file: File path where the combined grid image will be saved.
    """
    grid_rows, grid_cols = grid_shape
    img_height, img_width = image_size

    # Initialize an empty grid with dimensions (grid_height, grid_width)
    grid_image = np.zeros((grid_rows * img_height, grid_cols * img_width))

    # Loop through the images and place them in the appropriate position on the grid
    for idx, image in enumerate(images):
        row_idx = idx // grid_cols  # Calculate the row index
        col_idx = idx % grid_cols   # Calculate the column index
        
        # Convert the torch tensor to numpy and scale it back to 0-255 for image representation
        image_np = image.cpu().numpy() * 255.0  # Assuming the images are normalized to [0, 1]
        image_np = image_np.astype(np.uint8)    # Convert to uint8 format for PIL
        
        if image_np.ndim == 3 and image_np.shape[0] == 1:
            image_np = image_np[0]  # Remove the channel dimension (1, height, width) -> (height, width)

        # Resize the image to the target size using PIL's bicubic interpolation
        image_resized = Image.fromarray(image_np).resize((img_width, img_height), Image.BICUBIC)

        # Place the resized image into the grid
        grid_image[
            row_idx * img_height:(row_idx + 1) * img_height,
            col_idx * img_width:(col_idx + 1) * img_width
        ] = np.array(image_resized)

        # Print shape of the first image for debugging purposes
        if idx == 0:
            print("Resized image shape:", image_resized.size)
            print("Original image shape:", image.shape)

    # Save the final grid image to the specified output file
    grid_image_pil = Image.fromarray(grid_image.astype(np.uint8))
    grid_image_pil.save(output_file)


# Set parameters
input_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 300
z_dim = 2
dropout_prob = 0.1  # Dropout probability (adjustable)
learning_rate = 1e-4
batch_size = 64
n_epochs = 200
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the VAE model, optimizer, and move to the appropriate device
model = VAE(input_dim, hidden_dim, z_dim, dropout_prob).to(device)
encoder = model.encoder; decoder = model.decoder

# plot for ground-truth
images, _ = next(iter(test_loader))
save_image_grid(images, (8,8), (28,28), f'{results_dir}/ground_truth.png')


# Define the loss function (ELBO)
def loss_function(recon_x, x, mean, logvar):
    # Binary cross-entropy loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Kullback-Leibler divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return BCE + KLD, BCE, KLD

# Test the model
model.eval()

# Load the trained model
model.load_state_dict(torch.load('vae.pth'))

with torch.no_grad():
    test_loss = 0
    
    # plot
    plt.figure(figsize=(8, 6))
    base = plt.cm.get_cmap("Paired", 10)
    N = 10
    
    
    for i, (data, label) in enumerate(test_loader):
        data = data.view(-1, input_dim).to(device)
        recon_batch, mean, logvar = model(data)
        loss, _, _ = loss_function(recon_batch, data, mean, logvar)
        test_loss += loss.item()
        
        # plot for latent batch
        label = label.cpu().numpy()
        latent_batch, _, _ = model.encoder(data)
        
        latent_batch = latent_batch.cpu().numpy()
        
        # Create scatter plot for the 2D latent space
        plt.scatter(latent_batch[:, 0], latent_batch[:, 1], c=label, marker='o', edgecolor='none', cmap=base, alpha=0.5)
    
    
    # Add color bar and set ticks
    plt.colorbar(ticks=range(N))
    # Set plot limits and axes
    axes = plt.gca()
    axes.set_xlim([-4, 4])
    axes.set_ylim([-4, 4])
            
    # Add grid and save the figure
    plt.grid(True)    
    plt.savefig(f"{results_dir}/latent_scatter.png")
    
    print(f'Test Loss: {test_loss / len(test_loader.dataset)}')


# Generate sample from the latent space
n_img_width=20; n_img_height=20
n_tot_imgs = n_img_width*n_img_height

z_input = np.mgrid[-4:4:n_img_width*1j, -4:4:n_img_height*1j]
z_input = np.moveaxis(z_input, 0, -1)  # Move the first axis to the last
z_input = z_input.reshape([-1, 2])  # Reshape to shape (-1, 2) for latent space

# Convert numpy array to a torch tensor and move it to the device (e.g., GPU if available)
z_input = torch.tensor(z_input, dtype=torch.float32).to(device)  # Ensure to use the correct device

# Perform a forward pass through the decoder network to generate images
with torch.no_grad():  # Disable gradient calculation for inference
    generated_images = model.decoder(z_input)  # Assuming `model.decoder` is the decoder part of the VAE
    generated_images = generated_images.cpu().view(n_tot_imgs, 28, 28)
    plt.figure(figsize=(n_img_width, n_img_height))
    for i in range(n_tot_imgs):
        plt.subplot(n_img_width, n_img_height, i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.savefig(f'{results_dir}/vae_eval.png')

