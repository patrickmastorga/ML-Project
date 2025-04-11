import torch
import torch.nn as nn

from sys import stdout
from typing import Callable
from time import strftime

def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader | None,
    criterion: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = 'cpu',
    log_file = stdout,
    log_interval: int | None = 16,
) -> tuple[list[float], list[float]]:
    """
    Train a model using the given dataloaders, loss function, and optimizer.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for testing data (optional)
        criterion (Callable[[nn.Module, torch.Tensor], torch.Tensor]): Function which takes the model and batch of data and returns the loss. Must make sure the batch is sent to the same device as the model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs to train.
        device (str): Device to use for training ('cpu' or 'cuda'). Default is 'cpu'.
        log_file (file-like object): File-like object to log output.
        log_interval (int | None): Interval for logging training progress. If None, no logging is done.

    Returns:
        tuple: Lists of training and testing losses.
    """
    logging = log_interval is not None
    if not logging:
        log_interval = len(train_dataloader) * 2

    model.to(device=device)
    if logging:
        print(f'{strftime('%H:%M:%S')} Training device: {device}', file=log_file)

    training_losses = []
    validation_losses = [] if test_dataloader is not None else None
    for epoch in range(epochs):
        if not logging or log_file is not stdout:
            print(f'{strftime('%H:%M:%S')} TRAINING Epoch [{epoch+1}/{epochs}]', end='')
        else:
            print(f'{strftime('%H:%M:%S')} BEGIN TRAINING Epoch [{epoch+1}/{epochs}]', file=log_file)

        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            # training step
            optimizer.zero_grad()
            loss = criterion(model, batch, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # keep track of loss and epoch progress
            if batch_idx % log_interval == log_interval - 1:
                training_losses.append(running_loss / log_interval)
                running_loss = 0.0
                print(f'{strftime('%H:%M:%S')} TRAINING Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {training_losses[-1]}', file=log_file)

        if test_dataloader is None:
            continue
        # validation
        if logging:
            print(f'{strftime('%H:%M:%S')} BEGIN VALIDATION Epoch [{epoch+1}/{epochs}]', file=log_file)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # validation step
                loss = criterion(model, batch, device)
                running_loss += loss.item()

        validation_losses.append(running_loss / len(test_dataloader))
        if log_file is not stdout or not logging:
            print(f' Loss: {validation_losses[-1]}')
        if logging:
            print(f'{strftime('%H:%M:%S')} VALIDATION Epoch [{epoch+1}/{epochs}], Loss: {validation_losses[-1]}', file=log_file)
    
    if logging:
        print(f'{strftime('%H:%M:%S')} Training complete.', file=log_file)
    return training_losses, validation_losses