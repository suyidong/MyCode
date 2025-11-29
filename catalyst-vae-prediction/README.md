# Catalyst VAE Prediction

A Variational Autoencoder (VAE) implementation for catalyst property prediction and latent space exploration. This project provides tools for training VAE models on catalyst data, making predictions, and exploring the latent space to discover new catalyst candidates.


## Features

- **VAE-based Regression**: Combines variational autoencoding with regression for catalyst prediction
- **Latent Space Exploration**: Systematic traversal of latent space to discover optimal catalyst configurations
- **Farthest Point Sampling**: Intelligent data sampling for better model generalization
- **Comprehensive Evaluation**: Multiple metrics and visualization tools

## Project Structure

```plaintext
catalyst-vae-prediction/
├── src/ # Source code
│ ├── train.py # Model training
│ ├── predict.py # Predict from input features
│ ├── latent_exploration.py # Latent space exploration
│ ├── model.py # VAE model architecture
│ └── utils.py # Utility functions
├── data/ # Data directory
└── models/ # Trained models used in our manuscript
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/catalyst-vae-prediction.git
cd catalyst-vae-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/train.py
```

### Predict from input features
```bash
python src/predict.py
```

### Latent space exploration
```bash
python src/latent_exploration.py
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
  url = {https://github.com/suyidong/MyCode/tree/main/catalyst-vae-prediction},
}
```
Our manuscript:
DOI: 

## Support

For any questions or issues, please contact us at: dongsuyi@tju.edu.cn

We are happy to assist you with any problems you may encounter during use.

