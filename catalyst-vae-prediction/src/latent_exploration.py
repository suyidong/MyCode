import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import itertools

# Set random seeds for reproducibility
torch.manual_seed(1895)
np.random.seed(1895)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define improved VAE model (same architecture as during training)
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

    def decode(self, z):
        """Decode directly from latent variable z"""
        decoded = self.decoder(z)
        reconstructed = self.reconstruction(decoded)
        prediction = self.regressor(decoded)
        return prediction, reconstructed


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
    model.eval()

    # Load scalers
    x_scaler = checkpoint['x_scaler']
    y_scaler = checkpoint['y_scaler']

    return model, x_scaler, y_scaler


def traverse_latent_space_and_predict(model, x_scaler, y_scaler,
                                      latent_dim=5,
                                      points_per_dim=10,
                                      value_range=(-3, 3),
                                      target_range=(0, 0.3)):
    """
    Traverse latent space and make predictions, filtering results within specified target value range

    Parameters:
    - model: Trained VAE model
    - x_scaler: Feature scaler
    - y_scaler: Target value scaler
    - latent_dim: Latent space dimension
    - points_per_dim: Number of points to sample per dimension
    - value_range: Range of latent variable values
    - target_range: Target prediction range for filtering
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

        # Convert latent space point to tensor
        z = torch.FloatTensor([point]).to(device)

        # Decode to get prediction and reconstruction
        with torch.no_grad():
            pred, reconstructed = model.decode(z)

        # Inverse transform
        pred_np = pred.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)[0][0]
        reconstructed_orig = x_scaler.inverse_transform(reconstructed_np)[0]

        # Filter results within specified target range
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
    """Save results to Excel file"""

    # Prepare data
    latent_vectors = [result['latent_vector'] for result in results]
    predictions = [result['prediction'] for result in results]
    reconstructed_features = [result['reconstructed_features'] for result in results]

    # Create DataFrames
    latent_df = pd.DataFrame(latent_vectors, columns=[f'latent_dim_{i + 1}' for i in range(len(latent_vectors[0]))])
    pred_df = pd.DataFrame(predictions, columns=['prediction'])

    # Reconstructed features DataFrame
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


def main():
    # Load model
    model_path = 'vae_catalyst_model.pth'
    model, x_scaler, y_scaler = load_model(model_path)
    print("Model loaded successfully!")

    # Traverse latent space and filter results
    print("Starting latent space traversal...")
    results = traverse_latent_space_and_predict(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        latent_dim=5,
        points_per_dim=8,  # 8 points per dimension, total 8^5=32768 points
        value_range=(-1.5, 1.5),  # Latent variable value range
        target_range=(-0.3, 0.3)  # Filter prediction range
    )

    if results:
        # Save results to Excel
        df = save_results_to_excel(results)

        # Print some statistics
        print(f"\nResult Statistics:")
        print(f"Found {len(results)} qualified samples")
        print(f"Prediction range: {df['prediction'].min():.4f} - {df['prediction'].max():.4f}")
        print(f"Mean prediction: {df['prediction'].mean():.4f}")

        # Show first few results
        print(f"\nFirst 5 results:")
        print(df.head())
    else:
        print("No qualified samples found, please adjust search range or sampling density")


if __name__ == "__main__":
    main()
