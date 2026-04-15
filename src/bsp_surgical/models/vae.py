import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Simple 5-layer strided-conv encoder/decoder (original "small" backbone)
# ---------------------------------------------------------------------------
class _SimpleEncoder(nn.Module):
    def __init__(self, resolution: int, latent_dim: int):
        super().__init__()
        depth = int(math.log2(resolution)) - 2  # 128 -> 5, 64 -> 4
        channels = [3, 32, 64, 128, 256, 256][: depth + 1]
        layers: list[nn.Module] = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        flat = channels[-1] * 4 * 4
        self._final_channels = channels[-1]
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class _SimpleDecoder(nn.Module):
    def __init__(self, resolution: int, latent_dim: int):
        super().__init__()
        depth = int(math.log2(resolution)) - 2
        channels = [3, 32, 64, 128, 256, 256][: depth + 1]
        self._final_channels = channels[-1]
        self.fc = nn.Linear(latent_dim, channels[-1] * 4 * 4)
        layers: list[nn.Module] = []
        for in_c, out_c in zip(reversed(channels[1:]), reversed(channels[:-1])):
            layers += [nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)]
            if out_c != channels[0]:
                layers += [nn.ReLU(inplace=True)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, self._final_channels, 4, 4)
        return torch.sigmoid(self.deconv(h))


# ---------------------------------------------------------------------------
# ResNet-18 encoder + mirror upsample decoder (larger "resnet18" backbone)
# ---------------------------------------------------------------------------
class _ResNet18Encoder(nn.Module):
    """torchvision ResNet-18 stripped of its classification head.

    Input 128x128 -> output feature map 4x4x512 -> latent_dim.
    """

    def __init__(self, resolution: int, latent_dim: int, pretrained: bool = False):
        super().__init__()
        if resolution % 32 != 0:
            raise ValueError(f"ResNet-18 needs resolution % 32 == 0; got {resolution}")
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        spatial = resolution // 32
        flat = 512 * spatial * spatial
        self._flat = flat
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


def _up_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class _ResNet18Decoder(nn.Module):
    """Upsample-and-conv decoder roughly mirroring the ResNet-18 encoder's
    spatial reductions. Not a strict inverse — reconstructs 128x128."""

    def __init__(self, resolution: int, latent_dim: int):
        super().__init__()
        spatial = resolution // 32  # 4 for 128
        self._spatial = spatial
        self._channels = 512
        self.fc = nn.Linear(latent_dim, 512 * spatial * spatial)
        self.up = nn.Sequential(
            _up_block(512, 256),  # 4 -> 8
            _up_block(256, 128),  # 8 -> 16
            _up_block(128, 64),   # 16 -> 32
            _up_block(64, 32),    # 32 -> 64
            _up_block(32, 16),    # 64 -> 128
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor):
        h = self.fc(z).view(-1, self._channels, self._spatial, self._spatial)
        return torch.sigmoid(self.up(h))


# ---------------------------------------------------------------------------
# VAE front-end selects between the two backbones.
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        resolution: int = 128,
        backbone: str = "simple",
        pretrained: bool = False,
    ):
        super().__init__()
        if resolution not in (64, 128):
            raise ValueError(f"resolution must be 64 or 128, got {resolution}")
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.backbone = backbone

        if backbone == "simple":
            self.encoder = _SimpleEncoder(resolution, latent_dim)
            self.decoder = _SimpleDecoder(resolution, latent_dim)
        elif backbone == "resnet18":
            if resolution != 128:
                raise ValueError("resnet18 backbone requires resolution=128")
            self.encoder = _ResNet18Encoder(resolution, latent_dim, pretrained=pretrained)
            self.decoder = _ResNet18Decoder(resolution, latent_dim)
        else:
            raise ValueError(f"unknown backbone '{backbone}' (expected 'simple' or 'resnet18')")

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
