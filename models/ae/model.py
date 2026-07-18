import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Convolutional autoencoder encoder that maps images to one latent vector."""

    def __init__(self, z_size=64):
        """Initialize convolutional feature extraction and latent projection layers."""
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512 * 8 * 8, z_size)

    def forward(self, x):
        """Encode an image batch and return a VAE-compatible latent tuple."""
        z = self.fc(self.conv_layers(x))
        logvar = torch.zeros_like(z)
        return z, logvar


class Decoder(nn.Module):
    """Transposed-convolutional decoder that reconstructs images from latent vectors."""

    def __init__(self, z_size=64):
        """Initialize latent projection and image reconstruction layers."""
        super().__init__()
        self.fc = nn.Linear(z_size, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        """Decode latent vectors into reconstructed image tensors."""
        return self.deconv_layers(self.fc(z))


class BasicAE(nn.Module):
    """Small convolutional autoencoder wrapper for reconstruction baselines."""

    def __init__(self, z_size=64):
        """Initialize the encoder and decoder modules."""
        super().__init__()
        self.encoder = Encoder(z_size=z_size)
        self.decoder = Decoder(z_size=z_size)

    def forward(self, x):
        """Return the reconstruction and latent vector for an image batch."""
        z, _ = self.encoder(x)
        return self.decoder(z), z
