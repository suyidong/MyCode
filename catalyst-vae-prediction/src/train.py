import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import pairwise_distances

# Set font support for plots
plt.rcParams['axes.unicode_minus'] = False  # For negative sign display

# Set random seeds for reproducibility
torch.manual_seed(1895)
np.random.seed(1895)

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define improved VAE model
class ImprovedVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        super(ImprovedVAE, self).__init__()
        self.input_dim = input_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space mean and variance
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Reconstruction output layer
        self.reconstruction = nn.Linear(hidden_dims[0], input_dim)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dims[0], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        encoded = self.encoder(x)

        # Get latent space mean and variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decoding
        decoded = self.decoder(z)

        # Reconstruction
        reconstructed = self.reconstruction(decoded)

        # Prediction
        prediction = self.regressor(decoded)

        return prediction, reconstructed, mu, logvar


# Define loss function
def vae_loss(recon_pred, target, reconstructed, input_data, mu, logvar, beta=1.0, recon_weight=0.5):
    # Regression loss (MSE)
    mse_loss = nn.MSELoss()(recon_pred, target)

    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(reconstructed, input_data)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.mean(kl_loss)  # Average over batch

    return mse_loss + recon_weight * recon_loss + beta * kl_loss, mse_loss, recon_loss, kl_loss


# Train model
def train_vae(model, train_loader, optimizer, epochs, beta=1.0, recon_weight=0.5, scheduler=None):
    model.train()
    train_losses = []
    mse_losses = []
    recon_losses = []
    kl_losses = []
    learning_rates = []  # New: record learning rate changes

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

        # Adjust learning rate after each epoch
        if scheduler:
            scheduler.step()

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


# Model evaluation function - modified to evaluate any dataset
def evaluate_vae(model, X_data, y_data, x_scaler, y_scaler, dataset_name=""):
    model.eval()
    with torch.no_grad():
        pred, reconstructed, _, _ = model(X_data)

        # Calculate MSE for standardized data
        test_mse = nn.MSELoss()(pred, y_data).item()

        # Calculate reconstruction error
        recon_error = nn.MSELoss()(reconstructed, X_data).item()

        # Inverse transform predictions and true values
        pred_np = pred.cpu().numpy() if device == 'cuda' else pred.numpy()
        y_data_np = y_data.cpu().numpy() if device == 'cuda' else y_data.numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)
        y_data_orig = y_scaler.inverse_transform(y_data_np)

        # Calculate various evaluation metrics
        mse_orig = np.mean((pred_orig - y_data_orig) ** 2)
        mae_orig = np.mean(np.abs(pred_orig - y_data_orig))
        r2 = r2_score(y_data_orig, pred_orig)

        # Calculate deviation range for each point
        deviations = pred_orig.flatten() - y_data_orig.flatten()
        max_deviation = np.max(np.abs(deviations))
        mean_deviation = np.mean(np.abs(deviations))

        print(f"\n{dataset_name} Evaluation Results:")
        print(f"  MSE (Standardized Data): {test_mse:.4f}")
        print(f"  Reconstruction Error: {recon_error:.4f}")
        print(f"  MSE (Original Data): {mse_orig:.4f}")
        print(f"  MAE (Original Data): {mae_orig:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Maximum Deviation: {max_deviation:.4f}")
        print(f"  Mean Absolute Deviation: {mean_deviation:.4f}")
        print(f"  Deviation Range: [{deviations.min():.4f}, {deviations.max():.4f}]")

        return test_mse, recon_error, mse_orig, mae_orig, r2, pred_orig, y_data_orig, deviations


def main():
    # 1. Data preparation
    # Assuming your data is stored in a CSV file
    data = pd.read_excel("way/to/your/path")

    # Separate features and target
    X = data.drop('Target', axis=1).values
    y = data['Target'].values.reshape(-1, 1)
    X_vae = X[:, :2]

    # ---------- FPS Function ----------
    def fps_sample(X, ratio=0.8, metric='euclidean', random_state=None):
        """Farthest Point Sampling, return sampling indices"""
        np.random.seed(random_state)
        n_total = len(X)
        n_sample = int(n_total * ratio)

        if n_sample >= n_total:
            return np.arange(n_total)

        # 1. Randomly initialize first point
        idx_all = np.arange(n_total)
        select_idx = [np.random.choice(idx_all)]

        # 2. Initialize distance vector: shortest distance from each point to selected set
        dist = pairwise_distances(X, X[select_idx[-1]:select_idx[-1] + 1], metric=metric).ravel()
        dist[select_idx[-1]] = -1  # Mark selected points as -1

        while len(select_idx) < n_sample:
            # Select point with maximum distance (excluding selected points)
            farthest = np.argmax(dist)
            select_idx.append(farthest)
            dist[farthest] = -1  # Mark as selected

            # Update shortest distance
            if len(select_idx) < n_sample:
                new_dist = pairwise_distances(X, X[farthest:farthest + 1], metric=metric).ravel()
                dist = np.where(dist == -1, -1, np.minimum(dist, new_dist))

        return np.array(select_idx)

    # ---------- One-step FPS Sampling ----------
    fps_normalized = StandardScaler()
    X_normalized = fps_normalized.fit_transform(X)
    sample_ratio = 0.48
    fps_idx = fps_sample(X_normalized, ratio=sample_ratio, metric='euclidean', random_state=1895)
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[fps_idx] = True
    test_mask = ~train_mask

    # ---------- Fit scaler only on training set ----------
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_raw = X_vae[train_mask]
    y_train_raw = y[train_mask]
    X_test_raw = X_vae[test_mask]
    y_test_raw = y[test_mask]

    x_scaler.fit(X_train_raw)  # Use only training set
    y_scaler.fit(y_train_raw)

    # ---------- Transform both datasets ----------
    X_train = x_scaler.transform(X_train_raw)
    y_train = y_scaler.transform(y_train_raw)
    X_test = x_scaler.transform(X_test_raw)
    y_test = y_scaler.transform(y_test_raw)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. Set hyperparameters and train model
    input_dim = X_vae.shape[1]
    latent_dim = 5
    hidden_dims = [64, 32]  # Hidden layer dimensions

    model = ImprovedVAE(input_dim, latent_dim, hidden_dims).to(device)

    # Modified: Use AdamW optimizer instead of Adam
    optimizer = optim.AdamW(model.parameters(), lr=0.004)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.7)

    epochs = 1000
    beta = 0.01  # Reduce KL divergence weight coefficient, focus more on prediction accuracy
    recon_weight = 0.8  # Reconstruction loss weight

    # Train model
    print("Starting VAE training...")
    train_losses, mse_losses, recon_losses, kl_losses, learning_rates = train_vae(
        model, train_loader, optimizer, epochs, beta, recon_weight, scheduler
    )

    # 3. Evaluate model - evaluate both training and test sets
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)

    # Evaluate training set
    train_mse, train_recon_error, train_mse_orig, train_mae_orig, train_r2, train_pred_orig, train_y_orig, train_deviations = evaluate_vae(
        model, X_train_tensor, y_train_tensor, x_scaler, y_scaler, "Training Set"
    )

    # Evaluate test set
    test_mse, test_recon_error, test_mse_orig, test_mae_orig, test_r2, test_pred_orig, test_y_orig, test_deviations = evaluate_vae(
        model, X_test_tensor, y_test_tensor, x_scaler, y_scaler, "Test Set"
    )

    # 4. Visualize results
    plt.figure(figsize=(20, 12))

    # Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(train_losses, label='Total Loss')
    plt.plot(mse_losses, label='MSE Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(kl_losses, label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')

    # Training set predicted vs true values scatter plot
    plt.subplot(2, 4, 2)
    plt.scatter(train_y_orig, train_pred_orig, alpha=0.5, color='blue', label='Training Set')
    plt.plot([train_y_orig.min(), train_y_orig.max()],
             [train_y_orig.min(), train_y_orig.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Training Set Prediction Comparison (R² = {train_r2:.3f})')
    plt.legend()

    # Test set predicted vs true values scatter plot
    plt.subplot(2, 4, 3)
    plt.scatter(test_y_orig, test_pred_orig, alpha=0.5, color='green', label='Test Set')
    plt.plot([test_y_orig.min(), test_y_orig.max()],
             [test_y_orig.min(), test_y_orig.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test Set Prediction Comparison (R² = {test_r2:.3f})')
    plt.legend()

    # Error distribution comparison
    plt.subplot(2, 4, 4)
    plt.hist(train_deviations, bins=30, edgecolor='black', alpha=0.7, label='Training Set', color='blue')
    plt.hist(test_deviations, bins=30, edgecolor='black', alpha=0.7, label='Test Set', color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution Comparison')
    plt.legend()

    # Learning rate changes
    plt.subplot(2, 4, 5)
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Changes')
    plt.yscale('log')

    # Reconstruction error example
    plt.subplot(2, 4, 6)
    model.eval()
    with torch.no_grad():
        # Randomly select a few samples to show reconstruction effect
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        sample_data = X_test_tensor[sample_indices]
        _, reconstructed, _, _ = model(sample_data)

        # Inverse transform
        sample_orig = x_scaler.inverse_transform(sample_data.cpu().numpy())
        reconstructed_orig = x_scaler.inverse_transform(reconstructed.cpu().numpy())

        # Plot comparison of original features and reconstructed features
        for i in range(min(3, len(sample_indices))):  # Show at most 3 samples
            plt.plot(sample_orig[i], 'o-', label=f'Original Sample{i + 1}' if i == 0 else "")
            plt.plot(reconstructed_orig[i], 's--', label=f'Reconstructed Sample{i + 1}' if i == 0 else "")

        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        plt.title('Original Features vs Reconstructed Features')
        plt.legend()

    # Training set and test set RMSE comparison
    plt.subplot(2, 4, 7)
    train_rmse = np.sqrt(train_mse_orig)
    test_rmse = np.sqrt(test_mse_orig)
    datasets = ['Training Set', 'Test Set']
    rmse_scores = [train_rmse, test_rmse]
    colors = ['blue', 'green']
    bars = plt.bar(datasets, rmse_scores, color=colors, alpha=0.7)
    plt.ylabel('RMSE')
    plt.title('Training Set vs Test Set RMSE Comparison')
    # Display values on bars
    for bar, score in zip(bars, rmse_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    # MAE comparison
    plt.subplot(2, 4, 8)
    mae_scores = [train_mae_orig, test_mae_orig]
    bars = plt.bar(datasets, mae_scores, color=colors, alpha=0.7)
    plt.ylabel('MAE')
    plt.title('Training Set vs Test Set MAE Comparison')
    # Display values on bars
    for bar, score in zip(bars, mae_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('vae_catalyst_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 5. Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims
    }, 'vae_catalyst_model.pth')

    print("\nModel saved as 'vae_catalyst_model.pth'")


if __name__ == "__main__":
    main()
