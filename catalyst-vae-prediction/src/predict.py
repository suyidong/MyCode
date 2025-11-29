import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(1895)
np.random.seed(1895)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define improved VAE model (same complete structure as during training)
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
        # Encode
        encoded = self.encoder(x)

        # Get latent space mean and variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterization
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decoder(z)

        # Reconstruct
        reconstructed = self.reconstruction(decoded)

        # Predict
        prediction = self.regressor(decoded)

        return prediction, reconstructed, mu, logvar


def load_model(model_path):
    """Load trained model and scalers"""
    checkpoint = torch.load(model_path, map_location=device)

    # Create model instance
    model = ImprovedVAE(
        input_dim=checkpoint['input_dim'],
        latent_dim=checkpoint['latent_dim'],
        hidden_dims=checkpoint['hidden_dims']
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    # Load scalers
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']

    return model, x_scaler, y_scaler


def predict(model, x_scaler, y_scaler, X_new):
    """Make predictions using trained model"""
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


def main():
    # Load model and scalers
    model_path = 'vae_catalyst_model.pth'
    model, x_scaler, y_scaler = load_model(model_path)
    print("Model loaded successfully!")
    print(f"Input dimension: {x_scaler.n_features_in_}")

    # Example: Predict on new data
    n_features = x_scaler.n_features_in_
    new_data = np.random.randn(5, n_features)  # 5 new samples

    print("New data (original):")
    print(new_data)

    # Make predictions
    predictions, _ = predict(model, x_scaler, y_scaler, new_data)
    print("\nPredictions for new data:")
    print(predictions)

    # Example: Load data from CSV file for prediction
    try:
        # Replace with your actual file path
        # new_data_df = pd.read_csv('new_catalyst_data.csv')
        # new_data = new_data_df.values
        # predictions, reconstructions = predict(model, x_scaler, y_scaler, new_data)
        # print("\nPredictions for data loaded from file:")
        # print(predictions)
        pass
    except Exception as e:
        print(f"Error loading data from file: {e}")


if __name__ == "__main__":
    main()
