# imports
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
from pathlib import Path
from time import time

from mnist_vae import VAE, LinearEncoder, LinearDecoder, ConvDecoder, ConvEncoder, ELBOloss

from train import train_model
from utils import generate_new_images, plot_latent_space, generate_latent_space_traversal

def criterion(model, batch, device) -> torch.Tensor:
    """
    Criterion function for training the model. Computes the loss using the ELBO loss function.

    Args:
        model (nn.Module): The model to train.
        batch (torch.Tensor): A batch of data.

    Returns:
        torch.Tensor: The computed loss.
    """
    (images,) = batch
    images = images.to(device)
    reconstructed_x, mu, logvar = model(images)
    return ELBOloss(reconstructed_x, images, mu, logvar)

if __name__ == '__main__':
    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    LATENT_DIMS = 2
    MODEL_NAME = 'fmnist_sneaker_vae_linear_lg'

    # download MNIST dataset
    print('Loading dataset...')
    train_data = FashionMNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_data = FashionMNIST(root='./data', train=False, transform=ToTensor(), download=True)

    # filter dataset to only include class 9 (ankle boots)
    train_data_sneaker = TensorDataset(train_data.data[train_data.targets == 7].unsqueeze(1).float() / 255.0)
    test_data_sneaker = TensorDataset(test_data.data[test_data.targets == 7].unsqueeze(1).float() / 255.0)

    # define dataloaders for delivering batched data
    train_dataloader = DataLoader(dataset=train_data_sneaker, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(dataset=test_data_sneaker, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define models
    encoder = LinearEncoder(latent_dims=LATENT_DIMS, hidden_layer_size=1024).to(device=device)
    decoder = LinearDecoder(latent_dims=LATENT_DIMS, hidden_layer_size=1024).to(device=device)
    vae = VAE(encoder, decoder).to(device=device)

    # initialize optimizer
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)

    dir_path = Path('models') / MODEL_NAME
    dir_path.mkdir(parents=True, exist_ok=True)

    # train model
    print('Training VAE...')
    start_time = time()

    log_path = dir_path / "log.txt"
    with open(log_path, "w") as f:
        training_losses, validation_losses = train_model(
            model=vae,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=device,
            log_file=f,
        )

    end_time = time()
    training_time = end_time - start_time
    print('Training complete!')
    print(f"Training took {training_time:.2f} seconds")

    # save model state, optimizer state, and loss history
    torch.save(vae.state_dict(), dir_path / 'model.pth')
    torch.save({
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": training_losses,
        "val_losses": validation_losses
    }, dir_path / 'checkpoint.pth')

    # plot training loss
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.savefig(dir_path / 'training_loss.png')
    plt.clf()

    # plot the latent space (only works when LATENT_DIMS = 2)
    if LATENT_DIMS == 2:
        plot_latent_space(
            encoder=encoder,
            dataloader=test_dataloader,
            path=dir_path / 'latent_plot.png',
            batch_size=BATCH_SIZE,
            num_batches=6,
            device=device,
        )

        generate_latent_space_traversal(
            decoder=decoder,
            path=dir_path / 'latent_traversal.png',
            size=20,
            device=device,
        )

    #generate new images
    generate_new_images(
        decoder=decoder,
        latent_dims=LATENT_DIMS,
        path=dir_path / 'generated_images.png',
        device=device,
    )