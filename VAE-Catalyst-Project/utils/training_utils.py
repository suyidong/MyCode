import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def vae_loss(recon_pred, target, reconstructed, input_data, mu, logvar, beta=1.0, recon_weight=0.5):
    """
    Calculate VAE loss function.
    
    Args:
        recon_pred (Tensor): Regression prediction
        target (Tensor): Target values
        reconstructed (Tensor): Reconstructed input
        input_data (Tensor): Original input data
        mu (Tensor): Mean of latent distribution
        logvar (Tensor): Log variance of latent distribution
        beta (float): Weight for KL divergence
        recon_weight (float): Weight for reconstruction loss
        
    Returns:
        tuple: (total_loss, mse_loss, recon_loss, kl_loss)
    """
    # Regression loss (MSE)
    mse_loss = nn.MSELoss()(recon_pred, target)

    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(reconstructed, input_data)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)  # Average over batch

    total_loss = mse_loss + recon_weight * recon_loss + beta * kl_loss
    return total_loss, mse_loss, recon_loss, kl_loss

def evaluate_vae(model, X_data, y_data, x_scaler, y_scaler, dataset_name=""):
    """
    Evaluate VAE model performance.
    
    Args:
        model: Trained VAE model
        X_data (Tensor): Input features
        y_data (Tensor): Target values
        x_scaler: Feature scaler
        y_scaler: Target scaler
        dataset_name (str): Name of dataset for logging
        
    Returns:
        tuple: Evaluation metrics and results
    """
    model.eval()
    with torch.no_grad():
        pred, reconstructed, _, _ = model(X_data)

        # Calculate MSE on standardized data
        test_mse = nn.MSELoss()(pred, y_data).item()

        # Calculate reconstruction error
        recon_error = nn.MSELoss()(reconstructed, X_data).item()

        # Inverse transform predictions and targets
        pred_np = pred.cpu().numpy()
        y_data_np = y_data.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)
        y_data_orig = y_scaler.inverse_transform(y_data_np)

        # Calculate evaluation metrics
        mse_orig = np.mean((pred_orig - y_data_orig) ** 2)
        mae_orig = np.mean(np.abs(pred_orig - y_data_orig))
        r2 = r2_score(y_data_orig, pred_orig)

        # Calculate deviation statistics
        deviations = pred_orig.flatten() - y_data_orig.flatten()
        max_deviation = np.max(np.abs(deviations))
        mean_deviation = np.mean(np.abs(deviations))

        print(f"\n{dataset_name} Evaluation Results:")
        print(f"  MSE (standardized data): {test_mse:.4f}")
        print(f"  Reconstruction error: {recon_error:.4f}")
        print(f"  MSE (original data): {mse_orig:.4f}")
        print(f"  MAE (original data): {mae_orig:.4f}")
        print(f"  R² score: {r2:.4f}")
        print(f"  Maximum deviation: {max_deviation:.4f}")
        print(f"  Mean absolute deviation: {mean_deviation:.4f}")
        print(f"  Deviation range: [{deviations.min():.4f}, {deviations.max():.4f}]")

        return test_mse, recon_error, mse_orig, mae_orig, r2, pred_orig, y_data_orig, deviations
