# VAE Catalyst Project

A Variational Autoencoder (VAE) implementation for catalyst property prediction and latent space exploration. This project provides tools for training VAE models on catalyst data, making predictions, and exploring the latent space to discover new catalyst candidates.

## Project Structure
```plaintext
VAE-Catalyst-Project/
├── models/
│   ├── __init__.py
│   └── vae_model.py              # VAE model definition
├── utils/
│   ├── __init__.py
│   ├── data_utils.py             # Data processing utilities
│   └── training_utils.py         # Training utilities
├── training/
│   ├── __init__.py
│   └── train_vae.py              # Training script
├── prediction/
│   ├── __init__.py
│   ├── predict_one.py            # Predict from input features
│   └── predict_latent_exploration.py # Latent space exploration
├── config.py                     # Configuration file
└── main.py                       # Main entry point
```

## Features

- **VAE with Regression**: Combines variational autoencoder with regression head for property prediction
- **Farthest Point Sampling**: Advanced dataset splitting for better model generalization
- **Latent Space Exploration**: Systematic traversal of latent space to discover new catalyst candidates
- **Comprehensive Evaluation**: Multiple metrics and visualization tools for model assessment

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9.0+
- scikit-learn 1.0.0+
- pandas
- numpy
- matplotlib
- openpyxl

### Install Dependencies

```bash
pip install torch scikit-learn pandas numpy matplotlib openpyxl
```
## Usage

### Quick Start

```bash
# Train the model
python main.py --mode train
```

```bash
# Make predictions from input features
python main.py --mode predict
```

```bash
# Explore latent space for new candidates
python main.py --mode explore
```

### Training

Train a VAE model on your catalyst data:
```python
from training.train_vae import main as train_main
train_main()
```
Or use the command line:
```bash
python main.py --mode train
```

### Prediction
# Predict from input features
```python
from prediction.predict_one import load_model, predict

# Load trained model
model, x_scaler, y_scaler = load_model('vae_catalyst_model.pth')

# Make prediction
new_data = np.random.randn(1, n_features)  # Your input features
predictions, reconstructed = predict(model, x_scaler, y_scaler, new_data)
```
# Latent Space Exploration
Explore the latent space to discover new catalyst candidates with desired properties:
```python
from prediction.predict_latent_exploration import traverse_latent_space_and_predict

results = traverse_latent_space_and_predict(
    model=model,
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    latent_dim=5,
    points_per_dim=8,
    value_range=(-1.5, 1.5),
    target_range=(-0.3, 0.3)  # Filter for desired property range
)
```
### Configuration
Modify ```config.py``` to adjust model parameters and training settings:
```python
# Training configuration
TRAIN_CONFIG = {
    'random_seed': 1895,
    'batch_size': 32,
    'learning_rate': 0.004,
    'epochs': 1000,
    'latent_dim': 5,
    'hidden_dims': [64, 32],
    'beta': 0.01,  # KL divergence weight
    'recon_weight': 0.8,  # Reconstruction loss weight
}

# Data configuration
DATA_CONFIG = {
    'fps_ratio': 0.48,  # Farthest Point Sampling ratio
    'target_column': 'Target'
}
```
### Model Atchitecture
The Improved VAE model consists of:

- **Encoder**: Multi-layer perceptron with batch normalization and dropout

- **Latent Space**: Gaussian distribution with reparameterization trick

- **Decoder**: Symmetric to encoder for reconstruction

- **Regression Head**: Additional MLP for property prediction

## Data Format

### Input Data

- **Expected format**: Excel file (.xlsx)

- **Features**: Multiple catalyst descriptors

- **Target**: Single property column named 'Target'

### Example Data Structure

| Target | Feature1 | Feature2 | Feature3 | ... | 
|--------|----------|----------|----------|-----|
| 5.6    | 1.2      | 0.5      | 3.4      | ... |
| 7.2    | 2.1      | 1.2      | 4.1      | ... |

### Results
The training process generates:
- Loss curves (total, MSE, reconstruction, KL divergence)
  
- Prediction vs actual scatter plots
  
- Error distribution analysis
  
- Reconstruction examples
  
- Performance metrics (RMSE, MAE, R²)

## Citation
If you use this code in your research, please cite:

```bibtex
@software{vae_catalyst_project,
  title = {VAE Catalyst Project},
  author = {Dong, Suyi},
  year = {2025},
  url = {https://github.com/suyidong/MyCode/VAE-Catalyst-Project},
}
```

## Support
