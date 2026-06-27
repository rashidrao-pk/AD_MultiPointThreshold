import torch
import torch.nn as nn


# Encoder
class Encoder(nn.Module):
    """Convolutional VAE encoder that maps images to latent mean and log-variance."""

    def __init__(self, z_size=64):
        """Initialize convolutional layers and latent projection heads."""
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, z_size)
        self.fc_logvar = nn.Linear(512 * 8 * 8, z_size)

    def forward(self, x):
        """Encode an image batch into latent mean and log-variance tensors."""
        h = self.conv_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    """Transposed-convolutional VAE decoder that maps latent vectors to images."""

    def __init__(self, z_size=64):
        """Initialize latent projection and image reconstruction layers."""
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_size, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),      # 64x64 -> 128x128
            nn.Tanh()  # Output scaled between -1 and 1
        )

    def forward(self, z):
        """Decode latent vectors into reconstructed image tensors."""
        h = self.fc(z)
        h = self.deconv_layers(h)
        return h

# Define the model for blur/fake detection
class Discriminator(nn.Module):
    """CNN discriminator that scores images as real or reconstructed."""

    def __init__(self):
        """Initialize convolutional feature extraction and classifier layers."""
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )
    
    def forward(self, x):
        """Return discriminator logits for an image batch."""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
