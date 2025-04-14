import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

from mnist_vae import VAE, LinearEncoder, LinearDecoder
from utils import generate_latent_space_traversal, plot_images_grid

if __name__ == "__main__":
    MODEL_NAME = 'mnist_vae'
    LATENT_DIMS = 2

    dir_path = Path('models') / MODEL_NAME

    # download MNIST dataset
    print('Loading dataset...')
    test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, num_workers=0)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    encoder = LinearEncoder(latent_dims=LATENT_DIMS, hidden_layer_size=512).to(device=device)
    decoder = LinearDecoder(latent_dims=LATENT_DIMS, hidden_layer_size=512).to(device=device)
    vae = VAE(encoder, decoder).to(device=device)
    vae.load_state_dict(torch.load(dir_path / 'model.pth'))
    vae.eval()

    # plot original images
    images = next(iter(test_dataloader))[0].to(device)
    plot_images_grid(
        images=images,
        path=dir_path / 'original_images.png',
        size=(4, 4)
    )

    # plot reconstruciton
    with torch.no_grad():
        reconstructed, _, _ = vae(images)
    plot_images_grid(
        images=reconstructed,
        path=dir_path / 'reconstructed_images.png',
        size=(4, 4)
    )