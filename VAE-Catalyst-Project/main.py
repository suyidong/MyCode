"""
VAE Catalyst Project - Main Entry Point

This project implements a Variational Autoencoder with regression capabilities
for catalyst property prediction and latent space exploration.
"""

import argparse
from training.train_vae import main as train_main
from inference.predict_one import main as predict_main
from inference.predict_latent_exploration import main as explore_main

def main():
    """Main entry point for VAE Catalyst Project."""
    parser = argparse.ArgumentParser(description='VAE Catalyst Project')
    parser.add_argument('--mode', choices=['train', 'predict', 'explore'], 
                       required=True, help='Run mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_main()
    elif args.mode == 'predict':
        predict_main()
    elif args.mode == 'explore':
        explore_main()

if __name__ == "__main__":
    main()
