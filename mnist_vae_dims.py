# imports
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
from pathlib import Path
from time import time

from train import train_model
from utils import generate_new_images, plot_latent_space_with_labels, generate_latent_space_traversal

from mnist_vae import VAE, LinearEncoder, LinearDecoder, ConvDecoder, ConvEncoder, criterion

if __name__ == "__main__":
    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    # download MNIST dataset
    print('Loading dataset...')
    train_data = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

    # define dataloaders for delivering batched data
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=15)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_sizes = [512, 1024]
    for hidden_size in hidden_sizes:
        # create directory for saving models
        dir_path = Path('models') / f'mnist_vae_dims_{hidden_size}'
        dir_path.mkdir(parents=True, exist_ok=True)

        for latent_dims in [1, 2, 3, 4, 6, 10, 16, 24, 32]:
            model_name = f"vae_{latent_dims}d"

            # define models
            encoder = LinearEncoder(latent_dims=latent_dims, hidden_layer_size=hidden_size).to(device=device)
            decoder = LinearDecoder(latent_dims=latent_dims, hidden_layer_size=hidden_size).to(device=device)
            vae = VAE(encoder, decoder).to(device=device)

            # initialize optimizer
            optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)

            # train model
            print(f'Training {model_name}...')
            start_time = time()

            _, _ = train_model(
                model=vae,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=EPOCHS,
                device=device,
                log_interval=None
            )

            end_time = time()
            training_time = end_time - start_time
            print('Training complete!')
            print(f"Training took {training_time:.2f} seconds")

            # save model state, optimizer state, and loss history
            #torch.save(vae.state_dict(), dir_path / f'{model_name}.pth')

            #generate new images
            generate_new_images(
                decoder=decoder,
                latent_dims=latent_dims,
                path=dir_path / f'{model_name}_images.png',
                size=(10, 10),
                device=device,
            )
        
    print('Done!')