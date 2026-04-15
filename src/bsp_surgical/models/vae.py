import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    """Strided-conv encoder. Halves spatial dim five times for 128x128,
    four times for 64x64 (resolution / 2^depth = 4)."""

    def __init__(self, resolution: int, latent_dim: int):
        super().__init__()
        depth = int(math.log2(resolution)) - 2  # 128 -> 5, 64 -> 4
        channels = [3, 32, 64, 128, 256, 256]
        channels = channels[: depth + 1]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self._flat_dim = channels[-1] * 4 * 4
        self._final_channels = channels[-1]
        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    def __init__(self, resolution: int, latent_dim: int):
        super().__init__()
        depth = int(math.log2(resolution)) - 2
        channels = [3, 32, 64, 128, 256, 256]
        channels = channels[: depth + 1]
        self._final_channels = channels[-1]
        self.fc = nn.Linear(latent_dim, channels[-1] * 4 * 4)

        layers: list[nn.Module] = []
        for in_c, out_c in zip(reversed(channels[1:]), reversed(channels[:-1])):
            layers += [nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            # no ReLU after the final output layer
            if out_c != channels[0]:
                layers += [nn.ReLU(inplace=True)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self._final_channels, 4, 4)
        return torch.sigmoid(self.deconv(h))


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, resolution: int = 128):
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError(f"resolution must be 64 or 128, got {resolution}")
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.encoder = _Encoder(resolution, latent_dim)
        self.decoder = _Decoder(resolution, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
