import torch
import numpy as np
import pandas as pd
from models.vae_model import ImprovedVAE

def load_model(model_path):
    """
    Load trained model and scalers.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        tuple: (model, x_scaler, y_scaler)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def predict(model, x_scaler, y_scaler, X_new):
    """
    Make predictions using trained model.
    
    Args:
        model: Trained VAE model
        x_scaler: Feature scaler
        y_scaler: Target scaler
        X_new (ndarray): New input data
        
    Returns:
        tuple: (predictions, reconstructed_features)
    """
    model.eval()
    with torch.no_grad():
        # Standardize input
        X_scaled = x_scaler.transform(X_new)
        X_tensor = torch.FloatTensor(X_scaled).to(next(model.parameters()).device)

        # Predict
        pred, reconstructed, _, _ = model(X_tensor)

        # Inverse transform predictions and reconstructions
        pred_np = pred.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        pred_orig = y_scaler.inverse_transform(pred_np)
        reconstructed_orig = x_scaler.inverse_transform(reconstructed_np)

        return pred_orig, reconstructed_orig

def main():
    """Main prediction function."""
    # Load model
    model_path = 'vae_catalyst_model.pth'
    model, x_scaler, y_scaler = load_model(model_path)
    print("Model loaded successfully!")
    print(f"Input dimension: {x_scaler.n_features_in_}")

    # Example prediction
    n_features = x_scaler.n_features_in_
    new_data = np.random.randn(5, n_features)

    print("New data (original):")
    print(new_data)

    # Make predictions
    predictions, _ = predict(model, x_scaler, y_scaler, new_data)
    print("\nPrediction results for new data:")
    print(predictions)

if __name__ == "__main__":
    main()
