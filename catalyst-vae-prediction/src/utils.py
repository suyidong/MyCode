"""
Utility functions for VAE catalyst prediction model.
Includes training, evaluation, prediction, and data processing functions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, pairwise_distances
from torch.utils.data import DataLoader, TensorDataset


def vae_loss(recon_pred, target, reconstructed, input_data, mu, logvar, beta=1.0, recon_weight=0.5):
    """
    Calculate VAE loss combining regression, reconstruction, and KL divergence.
    
    Args:
        recon_pred: Regression predictions
        target: Target values
        reconstructed: Reconstructed input
        input_data: Original input data
        mu: Latent mean
        logvar: Latent log variance
        beta: KL divergence weight
        recon_weight: Reconstruction loss weight
        
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


def train_vae(model, train_loader, optimizer, epochs, beta=1.0, recon_weight=0.5, scheduler=None):
    """
    Train the VAE model.
    
    Args:
        model: VAE model instance
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        epochs: Number of training epochs
        beta: KL divergence weight
        recon_weight: Reconstruction loss weight
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

        # Adjust learning rate if scheduler provided
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


def evaluate_vae(model, X_data, y_data, x_scaler, y_scaler, dataset_name=""):
    """
    Evaluate VAE model performance.
    
    Args:
        model: Trained VAE model
        X_data: Input features tensor
        y_data: Target values tensor
        x_scaler: Feature scaler
        y_scaler: Target scaler
        dataset_name: Name of dataset for printing
        
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
        pred_np = pred.cpu().numpy() if next(model.parameters()).is_cuda else pred.numpy()
        y_data_np = y_data.cpu().numpy() if next(model.parameters()).is_cuda else y_data.numpy()

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
        print(f"  MSE (Standardized): {test_mse:.4f}")
        print(f"  Reconstruction Error: {recon_error:.4f}")
        print(f"  MSE (Original): {mse_orig:.4f}")
        print(f"  MAE (Original): {mae_orig:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Max Deviation: {max_deviation:.4f}")
        print(f"  Mean Absolute Deviation: {mean_deviation:.4f}")
        print(f"  Deviation Range: [{deviations.min():.4f}, {deviations.max():.4f}]")

        return test_mse, recon_error, mse_orig, mae_orig, r2, pred_orig, y_data_orig, deviations


def fps_sample(X, ratio=0.8, metric='euclidean', random_state=None):
    """
    Farthest Point Sampling for data subset selection.
    
    Args:
        X: Input data array
        ratio: Sampling ratio
        metric: Distance metric
        random_state: Random seed
        
    Returns:
        array: Indices of sampled points
    """
    np.random.seed(random_state)
    n_total = len(X)
    n_sample = int(n_total * ratio)

    if n_sample >= n_total:
        return np.arange(n_total)

    # Randomly initialize first point
    idx_all = np.arange(n_total)
    select_idx = [np.random.choice(idx_all)]

    # Initialize distance vector: shortest distance to selected set
    dist = pairwise_distances(X, X[select_idx[-1]:select_idx[-1] + 1], metric=metric).ravel()
    dist[select_idx[-1]] = -1  # Mark selected points as -1

    while len(select_idx) < n_sample:
        # Select point with maximum distance (excluding selected points)
        farthest = np.argmax(dist)
        select_idx.append(farthest)
        dist[farthest] = -1  # Mark as selected

        # Update shortest distances
        if len(select_idx) < n_sample:
            new_dist = pairwise_distances(X, X[farthest:farthest + 1], metric=metric).ravel()
            dist = np.where(dist == -1, -1, np.minimum(dist, new_dist))

    return np.array(select_idx)


def load_model(model_path, device):
    """
    Load trained model and scalers.
    
    Args:
        model_path: Path to saved model file
        device: Device to load model on
        
    Returns:
        tuple: (model, x_scaler, y_scaler)
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Create model instance
    model = ImprovedVAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        hidden_dims=checkpoint['hidden_dims']
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scalers
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']

    return model, x_scaler, y_scaler


def predict(model, x_scaler, y_scaler, X_new, device):
    """
    Make predictions using trained model.
    
    Args:
        model: Trained VAE model
        x_scaler: Feature scaler
        y_scaler: Target scaler
        X_new: New input data
        device: Device for computation
        
    Returns:
        tuple: (predictions, reconstructions)
    """
    model.eval()
    with torch.no_grad():
        # Standardize input
        X_scaled = x_scaler.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # Predict
        pred, reconstructed, _, _ = model(X_tensor)

        # Inverse transform predictions and reconstructions
        pred_np = pred.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)
        reconstructed_orig = x_scaler.inverse_transform(reconstructed_np)

        return pred_orig, reconstructed_orig


def traverse_latent_space_and_predict(model, x_scaler, y_scaler, device,
                                      latent_dim=5,
                                      points_per_dim=10,
                                      value_range=(-3, 3),
                                      target_range=(0, 0.3)):
    """
    Traverse latent space and make predictions, filtering results within target range.
    
    Args:
        model: Trained VAE model
        x_scaler: Feature scaler
        y_scaler: Target scaler
        device: Device for computation
        latent_dim: Latent space dimension
        points_per_dim: Points to sample per dimension
        value_range: Range for latent variable values
        target_range: Target prediction range for filtering
        
    Returns:
        list: Filtered results
    """
    # Generate latent space grid points
    linspace = np.linspace(value_range[0], value_range[1], points_per_dim)
    grid_points = list(itertools.product(linspace, repeat=latent_dim))

    print(f"Generated {len(grid_points)} latent space points...")

    results = []
    valid_count = 0

    for i, point in enumerate(grid_points):
        if i % 1000 == 0:
            print(f"Processing point {i}/{len(grid_points)}...")

        # Convert latent point to tensor
        z = torch.FloatTensor([point]).to(device)

        # Decode to get prediction and reconstruction
        with torch.no_grad():
            pred, reconstructed = model.decode(z)

        # Inverse transform
        pred_np = pred.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)[0][0]
        reconstructed_orig = x_scaler.inverse_transform(reconstructed_np)[0]

        # Filter results within target range
        if target_range[0] <= pred_orig <= target_range[1]:
            result = {
                'latent_vector': point,
                'prediction': pred_orig,
                'reconstructed_features': reconstructed_orig
            }
            results.append(result)
            valid_count += 1

    print(f"Found {valid_count} samples with predictions in range {target_range}")
    return results


def save_results_to_excel(results, filename='latent_space_results.xlsx'):
    """
    Save results to Excel file with summary information.
    
    Args:
        results: List of result dictionaries
        filename: Output filename
        
    Returns:
        DataFrame: Combined results DataFrame
    """
    # Prepare data
    latent_vectors = [result['latent_vector'] for result in results]
    predictions = [result['prediction'] for result in results]
    reconstructed_features = [result['reconstructed_features'] for result in results]

    # Create DataFrames
    latent_df = pd.DataFrame(latent_vectors, columns=[f'latent_dim_{i + 1}' for i in range(len(latent_vectors[0]))])
    pred_df = pd.DataFrame(predictions, columns=['prediction'])

    # Reconstruction features DataFrame
    n_features = len(reconstructed_features[0])
    recon_df = pd.DataFrame(reconstructed_features, columns=[f'feature_{i + 1}' for i in range(n_features)])

    # Combine all data
    final_df = pd.concat([latent_df, pred_df, recon_df], axis=1)

    # Save to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Filtered Results', index=False)

        # Add summary information
        summary_data = {
            'Total Samples': [len(results)],
            'Prediction Range': [f"{final_df['prediction'].min():.4f} - {final_df['prediction'].max():.4f}"],
            'Mean Prediction': [final_df['prediction'].mean()],
            'Standard Deviation': [final_df['prediction'].std()]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Results saved to {filename}")
    return final_df


def prepare_data_loaders(X_train, y_train, batch_size=32, shuffle=True):
    """
    Prepare DataLoader for training.
    
    Args:
        X_train: Training features
        y_train: Training targets
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader: Training data loader
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def create_optimizer_scheduler(model, lr=0.004, step_size=300, gamma=0.7):
    """
    Create optimizer and scheduler for training.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        step_size: Scheduler step size
        gamma: Scheduler gamma
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler
