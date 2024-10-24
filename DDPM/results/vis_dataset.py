import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from torchinfo import summary
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    
    train_data = datasets.MNIST(root = './data',
                                train=True,
                                download=True,
                                transform=data_transforms)
    test_data = datasets.MNIST(root = './data',
                                train=False,
                                download=True,
                                transform=data_transforms)

    # dataset
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    
    
    with torch.no_grad():
        fig, axes = plt.subplots(8, 8, figsize=(15, 15))
        
        # image
        for image, label in test_dl:
            image = image[:64]
            image = image.to(device)
            image = image.cpu().detach().numpy()
            
            label = label[:64]
            label = label.to(device)
            label = label.cpu().detach().numpy()
            break
        
        # plot
        for i in range(64):
            axes[i//8][i%8].set_title(label[i])
            axes[i//8][i%8].imshow(image[i][0])

    # save figs
    plt.tight_layout()
    plt.savefig(f"valid_mnist.png")
    plt.clf()