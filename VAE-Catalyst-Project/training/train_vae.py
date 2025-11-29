import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from models.vae_model import ImprovedVAE
from utils.training_utils import vae_loss, evaluate_vae


def train_vae(data_path, device, train_loader, X_train_tensor, y_train_tensor, 
              X_test_tensor, y_test_tensor, x_scaler, y_scaler, input_dim):
    """
    Train VAE model.
    
    Args:
        data_path (str): Path to data file
        device: Training device (CPU/GPU)
        train_loader: DataLoader for training data
        X_train_tensor: Training features tensor
        y_train_tensor: Training targets tensor
        X_test_tensor: Test features tensor
        y_test_tensor: Test targets tensor
        x_scaler: Feature scaler
        y_scaler: Target scaler
        input_dim (int): Input dimension
    """
    import config
    
    # Set model parameters from config
    latent_dim = config.TRAIN_CONFIG['latent_dim']
    hidden_dims = config.TRAIN_CONFIG['hidden_dims']
    learning_rate = config.TRAIN_CONFIG['learning_rate']
    epochs = config.TRAIN_CONFIG['epochs']
    beta = config.TRAIN_CONFIG['beta']
    recon_weight = config.TRAIN_CONFIG['recon_weight']
    scheduler_step = config.TRAIN_CONFIG['scheduler_step']
    scheduler_gamma = config.TRAIN_CONFIG['scheduler_gamma']

    # Initialize model
    model = ImprovedVAE(input_dim, latent_dim, hidden_dims).to(device)

    # Set optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Train model
    print("Starting VAE training...")
    train_losses, mse_losses, recon_losses, kl_losses, learning_rates = _train_model(
        model, train_loader, optimizer, epochs, beta, recon_weight, scheduler
    )

    # Evaluate model
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)

    # Evaluate on training and test sets
    train_results = evaluate_vae(model, X_train_tensor, y_train_tensor, x_scaler, y_scaler, "Training Set")
    test_results = evaluate_vae(model, X_test_tensor, y_test_tensor, x_scaler, y_scaler, "Test Set")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims
    }, config.INFERENCE_CONFIG['model_path'])

    print(f"\nModel saved as '{config.INFERENCE_CONFIG['model_path']}'")

def _train_model(model, train_loader, optimizer, epochs, beta=1.0, recon_weight=0.5, scheduler=None):
    """
    Internal training function.
    
    Args:
        model: VAE model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        epochs (int): Number of training epochs
        beta (float): KL divergence weight
        recon_weight (float): Reconstruction loss weight
        scheduler: Learning rate scheduler
        
    Returns:
        tuple: Training losses and learning rates
    """
    model.train()
    train_losses = []
    mse_losses = []
    recon_losses = []
    kl_losses = []
    learning_rates = []

    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        total_recon = 0
        total_kl = 0

        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred, reconstructed, mu, logvar = model(data)

            # Calculate loss
            loss, mse_loss, recon_loss, kl_loss = vae_loss(
                pred, target, reconstructed, data, mu, logvar, beta, recon_weight
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        # Adjust learning rate at end of epoch
        if scheduler:
            scheduler.step()

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        train_losses.append(avg_loss)
        mse_losses.append(avg_mse)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, MSE = {avg_mse:.4f}, '
                  f'Recon = {avg_recon:.4f}, KL = {avg_kl:.4f}, LR = {current_lr:.6f}')

    return train_losses, mse_losses, recon_losses, kl_losses, learning_rates
