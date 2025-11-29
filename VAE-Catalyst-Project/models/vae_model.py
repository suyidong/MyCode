import torch
import torch.nn as nn

class ImprovedVAE(nn.Module):
    """
    Improved Variational Autoencoder with regression head for catalyst prediction.
    
    Args:
        input_dim (int): Dimension of input features
        latent_dim (int): Dimension of latent space
        hidden_dims (list): List of hidden layer dimensions for encoder/decoder
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=[64, 32]):
        super(ImprovedVAE, self).__init__()
        self.input_dim = input_dim

        # Encoder layers
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

        # Decoder layers
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

        # Regression head for prediction
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dims[0], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (Tensor): Mean of the latent Gaussian
            logvar (Tensor): Log variance of the latent Gaussian
            
        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass of the VAE.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            tuple: (prediction, reconstructed, mu, logvar)
        """
        # Encode
        encoded = self.encoder(x)

        # Get latent space mean and variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        decoded = self.decoder(z)

        # Reconstruct
        reconstructed = self.reconstruction(decoded)

        # Predict
        prediction = self.regressor(decoded)

        return prediction, reconstructed, mu, logvar

    def decode(self, z):
        """
        Decode directly from latent variable z.
        
        Args:
            z (Tensor): Latent vector
            
        Returns:
            tuple: (prediction, reconstructed_features)
        """
        decoded = self.decoder(z)
        reconstructed = self.reconstruction(decoded)
        prediction = self.regressor(decoded)
        return prediction, reconstructed
