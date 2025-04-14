import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from celeba_vae_bernoulli import VAE, ConvEncoder, ConvDecoder, CelebATransform
from utils import plot_images_grid, generate_new_images

if __name__ == "__main__":
    MODEL_NAME = 'celeba_vae_64'
    LATENT_DIMS = 64

    dir_path = Path('models') / MODEL_NAME

    # download MNIST dataset
    print('Loading dataset...')
    test_data = CelebA(root='./data', split='test', target_type='attr', transform=CelebATransform(), download=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=0)

    image = test_data[3][0]
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(dir_path / 'sample_image.png')
    plt.clf()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    encoder = ConvEncoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    decoder = ConvDecoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    vae = VAE(encoder, decoder).to(device=device)
    vae.load_state_dict(torch.load(dir_path / 'model.pth'))
    vae.eval()

    #generate new images
    generate_new_images(
        decoder=decoder,
        latent_dims=LATENT_DIMS,
        path=dir_path / 'generated_images2.png',
        size=(3, 10),
        imgsize=(3, 4),
        device=device
    )

    # plot original images
    images = next(iter(test_dataloader))[0].to(device)
    plot_images_grid(
        images=images,
        path=dir_path / 'original_images.png',
        size=(4, 4),
        imgsize=(3, 4)
    )

    # plot reconstruciton
    with torch.no_grad():
        reconstructed, _, _ = vae(images)
    plot_images_grid(
        images=reconstructed,
        path=dir_path / 'reconstructed_images.png',
        size=(4, 4),
        imgsize=(3, 4)
    )