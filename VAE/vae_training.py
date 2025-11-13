import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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

# Define the loss function (ELBO)
def loss_function(recon_x, x, mean, logvar):
    # Binary cross-entropy loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Kullback-Leibler divergence loss
    # KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    KLD = 0.5 * (mean.pow(2) + logvar.exp().pow(2) - logvar + 1. + 1e-8).sum()
    
    return BCE - KLD, BCE, KLD

# Set parameters
input_dim = 28 * 28  # MNIST images are 28x28
hidden_dim = 300
z_dim = 2
dropout_prob = 0.1  # Dropout probability (adjustable)
learning_rate = 1e-4
batch_size = 128
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

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, input_dim).to(device)
        
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}, BCE: {bce.item() / len(train_loader.dataset)}, KLD: {kld.item() / len(train_loader.dataset)}')
    
    # Save results periodically
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            model.eval()
            sample = torch.randn(64, z_dim).to(device)
            generated_images = model.decoder(sample).cpu().view(64, 28, 28)
            plt.figure(figsize=(5, 5))
            for i in range(64):
                plt.subplot(8, 8, i+1)
                plt.imshow(generated_images[i], cmap='gray')
                plt.axis('off')
            plt.savefig(f'{results_dir}/epoch_{epoch + 1}.png')

# Test the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = data.view(-1, input_dim).to(device)
        recon_batch, mean, logvar = model(data)
        loss, _, _  = loss_function(recon_batch, data, mean, logvar).item()
        test_loss += loss
    
    print(f'Test Loss: {test_loss / len(test_loader.dataset)}')

# save the model
torch.save(model.state_dict(), 'vae.pth')