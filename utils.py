import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_new_images(decoder: nn.Module, latent_dims: int, path, size: int = 20, device = 'cpu'):
    """
    Generates a grid of images by sampling from the latent space of the decoder with a normal distribution.
    The grid is saved as a PNG file at the specified path.

    Args:
        decoder (nn.Module): The decoder model.
        path (pathlike): The path to save the generated image.
        size (int): The number of steps in each direction of the latent space.
        device: The device to use for computation (CPU or GPU).
    """
    decoder.to(device=device)
    decoder.eval()

    # generate new images
    z = torch.randn((size ** 2, latent_dims), device=device)
    with torch.no_grad():
        images = decoder(z)

    # check if images are greyscale or RGB
    greyscale = images.shape[1] == 1

    # plot images in grid
    fig, axes = plt.subplots(nrows=size, ncols=size, figsize=(5,5))
    for idx, ax in enumerate(axes.flat):
        if greyscale:
            ax.imshow(images[idx].squeeze().cpu().numpy(), cmap='gray')
        else:
            ax.imshow(images[idx].permute(1, 2, 0).cpu().numpy())
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_adjustable('box')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(path)
    plt.clf()

def plot_latent_space(encoder: nn.Module, dataloader: torch.utils.data.DataLoader, path, batch_size: int, num_batches:int = 16, device = 'cpu'):
    """
    Plots the latent space of the encoder by encoding the images from the dataloader.
    The plot is saved as a PNG file at the specified path.

    Args:
        encoder (nn.Module): The encoder model. Must have a two dimensional latent space.
        dataloader (torch.utils.data.DataLoader): Dataloader for testing data. Can either be a tensor of training images of a tuple with the tensor of training images as the first element.
        path (pathlike): The path to save the generated image.
        batch_size (int): The number of images in each batch.
        num_batches (int): The number of batches to process.
        device: The device to use for computation (CPU or GPU).
    """
    encoder.to(device=device)
    encoder.eval()

    all_z = np.empty((num_batches * batch_size, 2))

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch[0] if not isinstance(batch, torch.Tensor) else batch
            images = images.to(device=device)

            mu, logvar = encoder(images)
            epsilon = torch.randn_like(logvar)
            z = mu + torch.exp(logvar / 2) * epsilon

            all_z[batch_idx * batch_size : (batch_idx + 1) * batch_size] = z.cpu().numpy()

            if batch_idx == num_batches - 1:
                break

    plt.scatter(all_z[:, 0], all_z[:, 1])
    plt.title('Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig(path)
    plt.clf()

def plot_latent_space_with_labels(encoder: nn.Module, dataloader: torch.utils.data.DataLoader, path, num_classes, batch_size: int, num_batches:int = 16, device = 'cpu'):
    """
    Plots the latent space of the encoder by encoding the images from the dataloader, and colors them by their labels.
    The plot is saved as a PNG file at the specified path.

    Args:
        encoder (nn.Module): The encoder model. Must have a two dimensional latent space.
        dataloader (torch.utils.data.DataLoader): Dataloader for testing data. Must return a tuple of (images, labels) with labels in the range [0, num_labels - 1].
        path (pathlike): The path to save the generated image.
        num_classes (int): The number of classes in the dataset.
        batch_size (int): The number of images in each batch.
        num_batches (int): The number of batches to process.
        device: The device to use for computation (CPU or GPU).
    """
    encoder.to(device=device)
    encoder.eval()

    all_labels = np.empty(num_batches * batch_size)
    all_z = np.empty((num_batches * batch_size, 2))

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device=device)

            all_labels[batch_idx * batch_size : (batch_idx + 1) * batch_size] = labels.cpu().numpy()

            mu, logvar = encoder(images)
            epsilon = torch.randn_like(logvar)
            z = mu + torch.exp(logvar / 2) * epsilon

            all_z[batch_idx * batch_size : (batch_idx + 1) * batch_size] = z.cpu().numpy()

            if batch_idx == num_batches - 1:
                break

    colors = plt.cm.get_cmap('tab10', num_classes)
    for label in range(num_classes):
        indices = np.where(all_labels == label)
        plt.scatter(all_z[indices, 0], all_z[indices, 1], color=colors(label), label=str(label))

    plt.title('Latent Space')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.savefig(path)
    plt.clf()

def generate_latent_space_traversal(decoder: nn.Module, path, size:int = 20, device = 'cpu'):
    """
    Generates a grid of images by traversing the latent space of the decoder.
    The grid is saved as a PNG file at the specified path.

    Args:
        decoder (nn.Module): The decoder model. Must have a two dimensional latent space.
        path (pathlike): The path to save the generated image.
        size (int): The number of steps in each direction of the latent space.
        device: The device to use for computation (CPU or GPU).
    """
    decoder.to(device=device)
    decoder.eval()

    grid = np.linspace(0.05, 0.95, size)
    points = torch.tensor([[norm.ppf(x), norm.ppf(y)] for x in grid for y in grid], dtype=torch.float32, device=device)

    with torch.no_grad():
        images = decoder(points)

    fig, axes = plt.subplots(nrows=size, ncols=size, figsize=(5,5))

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_adjustable('box')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(path)
    plt.clf()