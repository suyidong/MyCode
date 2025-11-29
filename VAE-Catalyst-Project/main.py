"""
VAE Catalyst Project - Main Entry Point

This project implements a Variational Autoencoder with regression capabilities
for catalyst property prediction and latent space exploration.
"""

import argparse
import torch
import numpy as np
from training.train_vae import train_vae
from prediction.predict_one import predict_single, load_model
from prediction.predict_latent_exploration import explore_latent_space
from utils.data_utils import prepare_data
import config

def setup_environment():
    """Setup random seeds and device configuration."""
    torch.manual_seed(config.TRAIN_CONFIG['random_seed'])
    np.random.seed(config.TRAIN_CONFIG['random_seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def train_mode(args):
    """Execute training mode."""
    print("Starting VAE training...")
    
    device = setup_environment()
    
    # Prepare data
    (train_loader, X_train_tensor, y_train_tensor, X_test_tensor, 
     y_test_tensor, x_scaler, y_scaler, input_dim) = prepare_data(
        args.data_path,
        target_column=config.DATA_CONFIG['target_column'],
        fps_ratio=config.DATA_CONFIG['fps_ratio']
    )
    
    # Train model
    train_vae(
        data_path=args.data_path,
        device=device,
        train_loader=train_loader,
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        X_test_tensor=X_test_tensor,
        y_test_tensor=y_test_tensor,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_dim=input_dim
    )

def predict_mode(args):
    """Execute prediction mode."""
    print("Starting single prediction...")
    
    device = setup_environment()
    
    # Load model and make prediction
    model, x_scaler, y_scaler = load_model(args.model_path)
    
    # Example prediction with random data
    n_features = x_scaler.n_features_in_
    new_data = np.random.randn(5, n_features)
    
    print("New data (original):")
    print(new_data)
    
    predictions, _ = predict_single(model, x_scaler, y_scaler, new_data)
    print("\nPrediction results for new data:")
    print(predictions)

def explore_mode(args):
    """Execute latent space exploration mode."""
    print("Starting latent space exploration...")
    
    device = setup_environment()
    
    # Load model
    model, x_scaler, y_scaler = load_model(args.model_path)
    
    # Explore latent space
    results = explore_latent_space(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        latent_dim=config.TRAIN_CONFIG['latent_dim'],
        points_per_dim=config.INFERENCE_CONFIG['latent_points_per_dim'],
        value_range=config.INFERENCE_CONFIG['latent_value_range'],
        target_range=config.INFERENCE_CONFIG['target_range']
    )
    
    if results:
        print(f"Found {len(results)} valid samples")
        # Additional result processing can be added here
    else:
        print("No valid samples found.")

def main():
    """Main entry point for VAE Catalyst Project."""
    parser = argparse.ArgumentParser(description='VAE Catalyst Project')
    parser.add_argument('--mode', choices=['train', 'predict', 'explore'], 
                       required=True, help='Run mode')
    parser.add_argument('--data_path', type=str, 
                       default=config.DATA_CONFIG['default_data_path'],
                       help='Path to data file')
    parser.add_argument('--model_path', type=str, 
                       default=config.INFERENCE_CONFIG['model_path'],
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'explore':
        explore_mode(args)

if __name__ == "__main__":
    main()
