# imports
import torch
import torch.nn as nn
from torchvision.datasets import CelebA
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
from pathlib import Path
from time import time

from train import train_model
from utils import generate_new_images

# define models
class ConvEncoder(nn.Module):
    """
    Encoder for the Variational Autoencoder (VAE).
    Maps input images to a latent space using a linear layer and ReLU activation.
    """
    def __init__(self, latent_dims: int = 64, base_channels: int = 32, nonlinearity: nn.Module = nn.ReLU):
        """
        Args:
            latent_dims (int): Number of dimensions in the latent space.
            initial_channels (int): Number of channels in the first convolutional layer.
            nonlinearity (nn.Module): Nonlinear activation function to use.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=4, stride=2, padding=1),                 # (3, 64, 48) -> (base_channels, 32, 24)
            nn.BatchNorm2d(base_channels),
            nonlinearity(),
            nn.Conv2d(in_channels=base_channels, out_channels=2*base_channels, kernel_size=4, stride=2, padding=1),   # (base_channels, 32, 24) -> (2*base_channels, 16, 12)
            nn.BatchNorm2d(2*base_channels),
            nonlinearity(),
            nn.Conv2d(in_channels=2*base_channels, out_channels=4*base_channels, kernel_size=4, stride=2, padding=1), # (2*base_channels, 16, 12) -> (4*base_channels, 8, 6)
            nn.BatchNorm2d(4*base_channels),
            nonlinearity(),
            nn.Conv2d(in_channels=4*base_channels, out_channels=8*base_channels, kernel_size=4, stride=2, padding=1), # (4*base_channels, 8, 6) -> (8*base_channels, 4, 3)
            nn.Flatten(),
            nonlinearity(),
        )
        self.output1 = nn.Linear(in_features=base_channels*8*4*3, out_features=latent_dims)
        self.output2 = nn.Linear(in_features=base_channels*8*4*3, out_features=latent_dims)

    def forward(self, x):
        x = self.network(x)
        mu = self.output1(x)
        logvar = self.output2(x)
        return mu, logvar

class ConvDecoder(nn.Module):
    """
    Decoder for the Variational Autoencoder (VAE).
    Maps latent space samples back to the original image space using a linear layer and ReLU activation.
    """
    def __init__(self, latent_dims: int = 64, base_channels: int = 32, nonlinearity: nn.Module = nn.ReLU):
        """
        Args:
            latent_dims (int): Number of dimensions in the latent space.
            hidden_layer_size (int): Size of the hidden layer.
            initial_channels (int): Number of channels in the first convolutional layer.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=latent_dims, out_features=base_channels*8*4*3),
            nonlinearity(),
            nn.Unflatten(dim=1, unflattened_size=(base_channels*8, 4, 3)),
            nn.ConvTranspose2d(in_channels=base_channels*8, out_channels=base_channels*4, kernel_size=4, stride=2, padding=1), # (8*base_channels, 4, 3) -> (4*base_channels, 8, 6)
            nonlinearity(),
            nn.ConvTranspose2d(in_channels=base_channels*4, out_channels=base_channels*2, kernel_size=4, stride=2, padding=1), # (4*base_channels, 8, 6) -> (2*base_channels, 16, 12)
            nonlinearity(),
            nn.ConvTranspose2d(in_channels=base_channels*2, out_channels=base_channels, kernel_size=4, stride=2, padding=1),   # (2*base_channels, 16, 12) -> (base_channels, 32, 24)
            nonlinearity(),
            nn.ConvTranspose2d(in_channels=base_channels, out_channels=3, kernel_size=4, stride=2, padding=1),                 # (base_channels, 32, 24) -> (3, 64, 48)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    Combines the encoder and decoder networks.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, logvar = self.encoder(x)
        epsilon = torch.randn_like(logvar)
        z = mu + torch.exp(logvar / 2) * epsilon
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

# define loss function
def ELBOloss(reconstucted_x: torch.Tensor, original_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float=1.0, avg: bool=True) -> torch.Tensor:
    """
    Computes the negative evidence lower bound (ELBO) of the data log likelihood.

    Args:
        reconstucted_x (torch.Tensor): batch of reconstructed images
        original_x (torch.Tensor): batch of orignal images
        mu (torch.Tensor): batch of means of encoder output distribution
        logvar (torch.Tensor): batch of log variances of encoder output distribution
        beta (float): weight for KL divergence term
        avg (bool): whether to average the loss over the batch

    Returns:
        torch.Tensor: the negative evidence lower bound (ELBO) of the data log likelihood
    """
    # since decoder output is bernoilli, use binary cross entropy
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstucted_x, original_x, reduction='sum')

    # kl divergence between output distribution of encoder and standard normal
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    elbo = reconstruction_loss + beta * kl_div

    if avg:
        batch_size = original_x.size(0)
        return  elbo / batch_size
    else:
        return elbo
    
def criterion(model: nn.Module, batch: torch.Tensor, device: str) -> torch.Tensor:
    """
    Criterion function for training the model. Computes the loss using the ELBO loss function.

    Args:
        model (nn.Module): The model to train.
        batch (torch.Tensor): A batch of data.

    Returns:
        torch.Tensor: The computed loss.
    """
    images, _ = batch
    images = images.to(device)
    reconstructed_x, mu, logvar = model(images)
    return ELBOloss(reconstructed_x, images, mu, logvar, beta=1.0)

# define transform for dataset
class CelebATransform:
    def __call__(self, img):
        img = TF.crop(img, top=60, left=41, height=128, width=96)
        img = TF.resize(img, (64, 48))
        img = TF.to_tensor(img) # map images to [0, 1]
        return img

if __name__ == "__main__":
    # hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    LATENT_DIMS = 64
    MODEL_NAME = 'celeba_vae_64'

    # download MNIST dataset
    print('Loading dataset...')
    train_data = CelebA(root='./data', split='train', target_type='attr', transform=CelebATransform(), download=True)
    test_data = CelebA(root='./data', split='test', target_type='attr', transform=CelebATransform(), download=True)

    # define dataloaders for delivering batched data
    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=15)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define models
    encoder = ConvEncoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    decoder = ConvDecoder(latent_dims=LATENT_DIMS, base_channels=64).to(device=device)
    vae = VAE(encoder, decoder).to(device=device)

    # initialize optimizer
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)

    # create directory for saving model
    dir_path = Path('models') / MODEL_NAME
    dir_path.mkdir(parents=True, exist_ok=True)

    # train model
    print('Training VAE...')
    start_time = time()

    # log_path = dir_path / "log.txt"
    # with open(log_path, "w") as f:
    #     training_losses, validation_losses = train_model(
    #         model=vae,
    #         train_dataloader=train_dataloader,
    #         test_dataloader=test_dataloader,
    #         criterion=criterion,
    #         optimizer=optimizer,
    #         epochs=EPOCHS,
    #         device=device,
    #         log_file=f,
    #     )

    training_losses, validation_losses = train_model(
        model=vae,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        log_interval=64,
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

    #generate new images
    generate_new_images(
        decoder=decoder,
        latent_dims=LATENT_DIMS,
        path=dir_path / 'generated_images.png',
        size=8,
        figsize=(6, 8),
        device=device
    )