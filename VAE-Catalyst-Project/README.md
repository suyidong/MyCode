# VAE Catalyst Project

A Variational Autoencoder (VAE) implementation for catalyst property prediction and latent space exploration. This project provides tools for training VAE models on catalyst data, making predictions, and exploring the latent space to discover new catalyst candidates.

## Project Structure
VAE-Catalyst-Project/
├── models/
│ ├── init.py
│ └── vae_model.py # VAE model definition
├── utils/
│ ├── init.py
│ ├── data_utils.py # Data processing utilities
│ └── training_utils.py # Training utilities
├── training/
│ ├── init.py
│ └── train_vae.py # Training script
├── prediction/
│ ├── init.py
│ ├── predict_one.py # Predict from input features
│ └── predict_latent_exploration.py # Latent space exploration
├── config.py # Configuration file
└── main.py # Main entry point
