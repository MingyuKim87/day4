import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torchvision
from torchvision import transforms 
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim import Adam



def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def load_MNIST_transformed_dataset():
    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])
    
    train_data = torchvision.datasets.MNIST(root = './data',
                                train=True,
                                download=True,
                                transform=data_transforms)
    test_data = torchvision.datasets.MNIST(root = './data',
                                train=False,
                                download=True,
                                transform=data_transforms)
    
    return train_data, test_data

def load_StanfordCars_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train_data = torchvision.datasets.StanfordCars(root="./data", download=True, 
                                         transform=data_transform)

    test_data = torchvision.datasets.StanfordCars(root="./data", download=True, 
                                         transform=data_transform, split='test')
    
    return train_data, test_data

def show_StanfordCars_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    
    # transform
    image = reverse_transforms(image)
    
    return image
    
    
def show_MNIST_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    
    return image[0]
    

from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, time_emb_dim=32, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size, stride, padding)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, img_channel=3):
        super().__init__()
        image_channels = img_channel
        down_channels = (64, 128, 256, 512)
        up_channels = (512, 256, 128, 64)
        out_dim = img_channel
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim=time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        ups_convt2d = {
            "kernel_size":[3, 4, 4, 2],
            "stride" : [2, 2, 2, 2],
            "padding" : [1, 1, 1, 1],
        }
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        kernel_size=ups_convt2d["kernel_size"][i], \
                                        stride=ups_convt2d["stride"][i], \
                                        padding=ups_convt2d["padding"][i], \
                                        time_emb_dim=time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep=20):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
        
def get_loss(model, x_0, t):
    device = x_0.device
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(IMG_SIZE=64, device='cuda', dataset="stanford_cars", epoch=0):
    assert dataset in ["stanford_cars", "mnist"], "Dataset not implemented"
    img_size = IMG_SIZE
    if dataset == "stanford_cars":
        assert img_size==64, "Stanford Cars dataset only supports 64x64 images"
    elif dataset == "mnist":
        assert img_size==28, "MNIST dataset only supports 28x28 images"
    else:
        NotImplementedError("Dataset not implemented")
    
    # Sample noise
    img = torch.randn((1, 1, img_size, img_size), device=device)
    num_images = 10
    stepsize = int(T/num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            if dataset == "stanford_cars":
                plt_img = show_StanfordCars_tensor_image(img.detach().cpu())
            elif dataset == "mnist":
                plt_img = show_MNIST_tensor_image(img.detach().cpu())
            else:
                raise NotImplementedError("Dataset not implemented")
            
            axes[int(i/stepsize)].set_title(f"t={int(i)}")
            axes[int(i/stepsize)].imshow(plt_img)
            
    plt.tight_layout()
    plt.savefig(f"results/results_{dataset}_{epoch:02d}.png")
    plt.clf()            
        
##############
# PARAMETERS #
##############
# Dataset
MNIST_IMG_SIZE = 28
StanfordCars_IMG_SIZE = 64
BATCH_SIZE = 128

mnist_train_data, mnist_test_data = load_MNIST_transformed_dataset()
mnist_train_dl = DataLoader(mnist_train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
mnist_test_dl = DataLoader(mnist_test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

cars_train_data, cars_test_data = load_StanfordCars_transformed_dataset(StanfordCars_IMG_SIZE)
cars_train_dl = DataLoader(cars_train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
cars_test_dl = DataLoader(cars_test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Simulate forward diffusion
image = next(iter(mnist_train_dl))[0]
num_images = 10
stepsize = int(T/num_images)
fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    img, noise = forward_diffusion_sample(image, t)
    plt_img = show_MNIST_tensor_image(img)
    axes[int(idx/stepsize)].set_title(f"t={int(idx)}")
    axes[int(idx/stepsize)].imshow(plt_img)

# save figs
plt.tight_layout()
plt.savefig(f"valid_mnist_2.png")
plt.clf()

#########
# MODEL #
#########
model = SimpleUnet(img_channel=1)   
print("Num params: ", sum(p.numel() for p in model.parameters()))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 500
results = []

for epoch in range(epochs):
    for step, batch in enumerate(mnist_train_dl):
      # zero_grad
      optimizer.zero_grad()

      # inference
      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      
      # assign device
      imgs = batch[0].to(device); t = t.to(device)
      loss = get_loss(model, imgs, t) # batch[0] indicates imgs excluding labels
      
      # backprop
      loss.backward()
      
      # update
      optimizer.step()

      if epoch % 50 == 0 and step == 0:
        # save ckpt
        torch.save(model.state_dict(), f"results/model_mnist.pt")

        # results 
        result = dict({"Epoch":epoch, "step": f"{step:03d}", "Loss" : f"{loss.item()}"})
        results.append(result)
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        
        # generate imgs
        sample_plot_image(
            IMG_SIZE=MNIST_IMG_SIZE,
            device=device,
            dataset="mnist",
            epoch=epoch
        )

# save results
results_df = pd.DataFrame(results)
results_df.to_csv("results/results_mnist.csv", index=False)