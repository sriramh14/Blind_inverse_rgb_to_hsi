"""Residual latent diffusion for NTIRE 2022 RGB-to-HSI reconstruction.

The complete model is intentionally kept in one file, matching the compact
layout of the user's reference DPS repository.

Stages
------
1. Train a high-fidelity 31-band HSI autoencoder.
2. Freeze it and train an RGB latent initializer plus HSI->RGB forward adapter.
3. Freeze both and train diffusion on the residual latent

       residual = standardized_HSI_latent - RGB_latent.

At inference DDIM predicts only this residual. Optional decoded RGB consistency
acts as a DPS-style correction in latent space.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ModelConfig:
    num_bands: int = 31
    hsi_min: float = 0.0
    hsi_max: float = 1.0

    ae_base_channels: int = 64
    latent_channels: int = 16
    rgb_base_channels: int = 48
    diffusion_base_channels: int = 96
    diffusion_channel_mults: Tuple[int, int, int] = (1, 2, 4)
    time_embedding_dim: int = 384
    attention_heads: int = 4
    dropout: float = 0.0

    diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"
    linear_beta_start: float = 1e-4
    linear_beta_end: float = 2e-2
    sampling_steps: int = 25
    ddim_eta: float = 0.0

    physics_guidance_scale: float = 0.03
    normalize_guidance: bool = True
    physics_blur_kernel: int = 5
    clip_denoised: bool = True

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelConfig":
        allowed = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in allowed}
        if "diffusion_channel_mults" in filtered:
            filtered["diffusion_channel_mults"] = tuple(filtered["diffusion_channel_mults"])
        return cls(**filtered)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# HELPERS
# =============================================================================


def _group_count(channels: int, maximum: int = 32) -> int:
    # Keep at least two channels per group. This remains valid when the deepest
    # latent feature is 1x1 and the physical batch size is one; using one
    # channel per group would make GroupNorm receive only a single value.
    upper = min(maximum, max(channels // 2, 1))
    for groups in reversed(range(1, upper + 1)):
        if channels % groups == 0:
            return groups
    return 1


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-8)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).clamp(max=0.999)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    mode = "reflect" if height > pad_h and width > pad_w else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (pad_h, pad_w)


def _remove_padding(x: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = padding
    height = x.shape[-2] - pad_h if pad_h else x.shape[-2]
    width = x.shape[-1] - pad_w if pad_w else x.shape[-1]
    return x[..., :height, :width]


def _gaussian_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        raise ValueError("physics_blur_kernel must be odd")
    sigma = max(kernel_size / 6.0, 0.5)
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords = coords - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    weight = kernel_2d.expand(x.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(x, weight, padding=kernel_size // 2, groups=x.shape[1])


# =============================================================================
# COMMON BLOCKS
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.time_projection = nn.Linear(time_dim, out_channels * 2) if time_dim else None

    def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.norm2(hidden)
        if self.time_projection is not None:
            if time_embedding is None:
                raise ValueError("time_embedding is required")
            scale, shift = self.time_projection(F.silu(time_embedding)).chunk(2, dim=1)
            hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return hidden + residual


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        return self.conv(F.interpolate(x, size=target_size, mode="nearest"))


class LinearSpatialAttention(nn.Module):
    """Linear-complexity spatial attention used only at the latent bottleneck."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.dim_head = channels // heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [
            tensor.reshape(batch, self.heads, self.dim_head, height * width)
            for tensor in (q, k, v)
        ]
        q = q.softmax(dim=2)
        k = k.softmax(dim=3)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        attended = torch.einsum("bhde,bhdn->bhen", context, q)
        attended = attended.reshape(batch, channels, height, width)
        return x + self.to_out(attended)


# =============================================================================
# STAGE 1: HSI AUTOENCODER
# =============================================================================


class HSIEncoder(nn.Module):
    def __init__(self, bands: int, : int, latent_channels: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(bands, base_channels, 3, padding=1)
        self.level1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.down1 = Downsample(base_channels, base_channels // 2)
        self.level2 = nn.Sequential(
            ResidualBlock(base_channels //2, base_channels // 2),
            ResidualBlock(base_channels // 2, base_channels // 2),
        )
        self.down2 = Downsample(base_channels // 2, base_channels // 4)
        self.level3 = nn.Sequential(
            ResidualBlock(base_channels // 4, base_channels // 4),
            ResidualBlock(base_channels // 4, base_channels // 4),
        )
        self.to_latent = nn.Conv2d(base_channels // 4, latent_channels, 3, padding=1)

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
       #Add skip connections
        x = self.stem(hsi) + self.level1(self.stem(hsi))
        x = self.level2(self.down1(x)) + self.down1(x)
        x = self.level3(self.down2(x)) + self.down2(x)
        return self.to_latent(x)


class HSIDecoder(nn.Module):
    def __init__(self, bands: int, base_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.from_latent = nn.Conv2d(latent_channels, base_channels // 4, 3, padding=1)
        self.level3 = nn.Sequential(
            ResidualBlock(base_channels // 4, base_channels // 4),
            ResidualBlock(base_channels // 4, base_channels // 4),
        )
        self.up2 = Upsample(base_channels // 4, base_channels // 2)
        self.level2 = nn.Sequential(
            ResidualBlock(base_channels // 2, base_channels // 2),
            ResidualBlock(base_channels // 2, base_channels // 2),
        )
        self.up1 = Upsample(base_channels // 2, base_channels)
        self.level1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.output = nn.Conv2d(base_channels, bands, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
       #Added skip connections
        x = self.level3(self.from_latent(latent)) + self.from_latent(latent)
        x = self.level2(self.up2(x, (x.shape[-2] * 2, x.shape[-1] * 2))) + self.up2(x, (x.shape[-2] * 2, x.shape[-1] * 2))
        x = self.level1(self.up1(x, (x.shape[-2] * 2, x.shape[-1] * 2))) + self.up1(x, (x.shape[-2] * 2, x.shape[-1] * 2))
        #return torch.sigmoid(self.output(x))
        return self.output(x)


class SpectralAutoencoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = HSIEncoder(
            config.num_bands, config.ae_base_channels, config.latent_channels
        )
        self.decoder = HSIDecoder(
            config.num_bands, config.ae_base_channels, config.latent_channels
        )

    def encode(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.encoder(hsi)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(hsi)
        return self.decode(latent), latent


# =============================================================================
# STAGE 2: RGB INITIALIZER AND FORWARD OPERATOR
# =============================================================================


class RGBLatentInitializer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.rgb_base_channels
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.level1 = nn.Sequential(ResidualBlock(base, base), ResidualBlock(base, base))
        self.down1 = Downsample(base, base // 2)
        self.level2 = nn.Sequential(
            ResidualBlock(base // 2, base // 2), ResidualBlock(base // 2, base // 2)
        )
        self.down2 = Downsample(base // 2, base // 4)
        self.level3 = nn.Sequential(
            ResidualBlock(base // 4, base // 4), ResidualBlock(base // 4, base // 4)
        )
        self.output = nn.Conv2d(base // 4, config.latent_channels, 3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.level1(self.stem(rgb)) + self.stem(rgb)
        x = self.level2(self.down1(x)) + self.down1(x)
        x = self.level3(self.down2(x)) + self.down2(x)
        return self.output(x)


class LearnedHSIToRGBOperator(nn.Module):
    """Positive spectral projection plus a small residual camera/ISP adapter."""

    def __init__(self, num_bands: int) -> None:
        super().__init__()
        wavelengths = torch.linspace(400.0, 700.0, num_bands)
        centers = torch.tensor([610.0, 545.0, 460.0]).view(3, 1)
        widths = torch.tensor([48.0, 42.0, 38.0]).view(3, 1)
        response = torch.exp(-0.5 * ((wavelengths.view(1, -1) - centers) / widths).square())
        response = response / response.sum(dim=1, keepdim=True)
        self.raw_response = nn.Parameter(_inverse_softplus(response))
        self.raw_gain = nn.Parameter(_inverse_softplus(torch.ones(3)))
        self.bias = nn.Parameter(torch.zeros(3))
        self.isp = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        self.raw_residual_scale = nn.Parameter(torch.tensor(-3.0))

    def normalized_response(self) -> torch.Tensor:
        response = F.softplus(self.raw_response) + 1e-8
        return response / response.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def matrix(self) -> torch.Tensor:
        return F.softplus(self.raw_gain)[:, None] * self.normalized_response()

    def linear_projection(self, hsi: torch.Tensor) -> torch.Tensor:
        weight = self.matrix().to(device=hsi.device, dtype=hsi.dtype)
        weight = weight[:, :, None, None]
        return F.conv2d(hsi, weight, self.bias.to(dtype=hsi.dtype))

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        linear = self.linear_projection(hsi)
        base = linear.clamp(1e-4, 1.0 - 1e-4)
        residual_scale = torch.sigmoid(self.raw_residual_scale)
        corrected_logit = torch.logit(base) + residual_scale * self.isp(base)
        return torch.sigmoid(corrected_logit)

    def smoothness_regularizer(self) -> torch.Tensor:
        response = self.normalized_response()
        second = response[:, 2:] - 2.0 * response[:, 1:-1] + response[:, :-2]
        return second.square().mean()


# =============================================================================
# STAGE 3: CONDITIONAL RESIDUAL LATENT DENOISER
# =============================================================================


class ConditionalResidualUNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        mults = config.diffusion_channel_mults
        if len(mults) != 3:
            raise ValueError("diffusion_channel_mults must contain exactly three values")
        c0, c1, c2 = [config.diffusion_base_channels * int(mult) for mult in mults]
        time_dim = config.time_embedding_dim
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(config.diffusion_base_channels),
            nn.Linear(config.diffusion_base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(config.latent_channels * 2, c0, 3, padding=1)
        self.enc0a = ResidualBlock(c0, c0, time_dim, config.dropout)
        self.enc0b = ResidualBlock(c0, c0, time_dim, config.dropout)
        self.down0 = Downsample(c0, c1)
        self.enc1a = ResidualBlock(c1, c1, time_dim, config.dropout)
        self.enc1b = ResidualBlock(c1, c1, time_dim, config.dropout)
        self.down1 = Downsample(c1, c2)
        self.mid1 = ResidualBlock(c2, c2, time_dim, config.dropout)
        self.attention = LinearSpatialAttention(c2, config.attention_heads)
        self.mid2 = ResidualBlock(c2, c2, time_dim, config.dropout)
        self.up1 = Upsample(c2, c1)
        self.dec1a = ResidualBlock(c1 + c1, c1, time_dim, config.dropout)
        self.dec1b = ResidualBlock(c1, c1, time_dim, config.dropout)
        self.up0 = Upsample(c1, c0)
        self.dec0a = ResidualBlock(c0 + c0, c0, time_dim, config.dropout)
        self.dec0b = ResidualBlock(c0, c0, time_dim, config.dropout)
        self.output = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            nn.SiLU(),
            nn.Conv2d(c0, config.latent_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
        rgb_latent: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_residual.shape != rgb_latent.shape:
            raise ValueError(
                f"noisy_residual and rgb_latent must match, got "
                f"{noisy_residual.shape} and {rgb_latent.shape}"
            )
        time_embedding = self.time_mlp(timesteps)
        x = self.stem(torch.cat((noisy_residual, rgb_latent), dim=1))
        skip0 = self.enc0b(self.enc0a(x, time_embedding), time_embedding)
        x = self.down0(skip0)
        skip1 = self.enc1b(self.enc1a(x, time_embedding), time_embedding)
        x = self.down1(skip1)
        x = self.mid2(self.attention(self.mid1(x, time_embedding)), time_embedding)
        x = self.up1(x, skip1.shape[-2:])
        x = self.dec1b(self.dec1a(torch.cat((x, skip1), dim=1), time_embedding), time_embedding)
        x = self.up0(x, skip0.shape[-2:])
        x = self.dec0b(self.dec0a(torch.cat((x, skip0), dim=1), time_embedding), time_embedding)
        return self.output(x)


# =============================================================================
# COMPLETE MODEL
# =============================================================================


class LatentDiffusionRGB2HSI(nn.Module):
    required_spatial_multiple = 16

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.autoencoder = SpectralAutoencoder(self.config)
        self.rgb_initializer = RGBLatentInitializer(self.config)
        self.forward_operator = LearnedHSIToRGBOperator(self.config.num_bands)
        self.denoiser = ConditionalResidualUNet(self.config)

        if self.config.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.config.diffusion_timesteps)
        elif self.config.beta_schedule == "linear":
            betas = linear_beta_schedule(
                self.config.diffusion_timesteps,
                self.config.linear_beta_start,
                self.config.linear_beta_end,
            )
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer(
            "latent_mean", torch.zeros(1, self.config.latent_channels, 1, 1)
        )
        self.register_buffer(
            "latent_std", torch.ones(1, self.config.latent_channels, 1, 1)
        )

    def set_latent_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected = (1, self.config.latent_channels, 1, 1)
        if tuple(mean.shape) != expected or tuple(std.shape) != expected:
            raise ValueError(f"Latent statistics must have shape {expected}")
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.to(self.latent_std).clamp_min(1e-6))

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std.clamp_min(1e-6)

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std.clamp_min(1e-6) + self.latent_mean

    def stage1(self, hsi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.autoencoder(hsi)

    def stage2(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_latent = self.rgb_initializer(rgb)
        hsi = self.autoencoder.decode(self.denormalize_latent(rgb_latent))
        return hsi, rgb_latent

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = (
            _extract(self.sqrt_alpha_bars, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape) * noise
        )
        return noisy, noise

    def predict_x0_from_epsilon(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = _extract(self.sqrt_alpha_bars, timesteps, noisy.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alpha_bars, timesteps, noisy.shape)
        return (noisy - sqrt_one_minus * epsilon) / sqrt_alpha.clamp_min(1e-8)

    def diffusion_training_outputs(
        self,
        rgb: torch.Tensor,
        hsi: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            hsi_latent = self.normalize_latent(self.autoencoder.encode(hsi))
            rgb_latent = self.rgb_initializer(rgb)
            clean_residual = hsi_latent - rgb_latent
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.diffusion_timesteps,
                (hsi.shape[0],),
                device=hsi.device,
                dtype=torch.long,
            )
        noisy_residual, noise = self.q_sample(clean_residual, timesteps, noise)
        predicted_noise = self.denoiser(noisy_residual, timesteps, rgb_latent)
        predicted_residual = self.predict_x0_from_epsilon(
            noisy_residual, timesteps, predicted_noise
        )
        return {
            "timesteps": timesteps,
            "noise": noise,
            "predicted_noise": predicted_noise,
            "clean_residual": clean_residual,
            "predicted_residual": predicted_residual,
            "rgb_latent": rgb_latent,
        }

    def sampling_timesteps(self, steps: int) -> Sequence[int]:
        if not 1 <= steps <= self.config.diffusion_timesteps:
            raise ValueError(
                f"steps must be in [1,{self.config.diffusion_timesteps}], got {steps}"
            )
        values = torch.linspace(0, self.config.diffusion_timesteps - 1, steps)
        values = values.round().long().tolist()
        return list(reversed(list(dict.fromkeys(values))))

    def ddim_step(
        self,
        noisy: torch.Tensor,
        epsilon: torch.Tensor,
        timestep: int,
        previous_timestep: int,
        eta: float,
    ) -> torch.Tensor:
        batch = noisy.shape[0]
        t = torch.full((batch,), timestep, device=noisy.device, dtype=torch.long)
        clean = self.predict_x0_from_epsilon(noisy, t, epsilon)
        alpha_t = self.alpha_bars[timestep].to(noisy)
        alpha_previous = (
            self.alpha_bars[previous_timestep].to(noisy)
            if previous_timestep >= 0
            else torch.ones((), device=noisy.device, dtype=noisy.dtype)
        )
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_previous) / (1.0 - alpha_t).clamp_min(1e-12))
            * (1.0 - alpha_t / alpha_previous.clamp_min(1e-12))
        ).clamp_min(0.0)
        direction = torch.sqrt(
            (1.0 - alpha_previous - sigma.square()).clamp_min(0.0)
        ) * epsilon
        stochastic = sigma * torch.randn_like(noisy) if eta > 0 else 0.0
        return torch.sqrt(alpha_previous) * clean + direction + stochastic

    @torch.enable_grad()
    def sample_residual(
        self,
        rgb_latent: torch.Tensor,
        observed_rgb: Optional[torch.Tensor] = None,
        sampling_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        steps = int(sampling_steps or self.config.sampling_steps)
        scale = self.config.physics_guidance_scale if guidance_scale is None else float(guidance_scale)
        if initial_noise is not None:
            if initial_noise.shape != rgb_latent.shape:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} does not match "
                    f"latent shape {tuple(rgb_latent.shape)}"
                )
            residual = initial_noise.to(device=rgb_latent.device, dtype=rgb_latent.dtype)
        else:
            try:
                residual = torch.randn(
                    rgb_latent.shape,
                    device=rgb_latent.device,
                    dtype=rgb_latent.dtype,
                    generator=generator,
                )
            except TypeError:
                residual = torch.randn_like(rgb_latent)
        sequence = self.sampling_timesteps(steps)
        guidance_enabled = observed_rgb is not None and scale > 0

        for index, timestep in enumerate(sequence):
            previous = sequence[index + 1] if index + 1 < len(sequence) else -1
            time_batch = torch.full(
                (residual.shape[0],), timestep, device=residual.device, dtype=torch.long
            )
            if guidance_enabled:
                residual = residual.detach().requires_grad_(True)
                epsilon = self.denoiser(residual, time_batch, rgb_latent)
                residual_x0 = self.predict_x0_from_epsilon(residual, time_batch, epsilon)
                latent = self.denormalize_latent(rgb_latent + residual_x0)
                hsi_hat = self.autoencoder.decode(latent)
                rgb_hat = self.forward_operator(hsi_hat)
                loss = F.smooth_l1_loss(
                    _gaussian_blur(rgb_hat, self.config.physics_blur_kernel),
                    _gaussian_blur(observed_rgb, self.config.physics_blur_kernel),
                )
                gradient = torch.autograd.grad(loss, residual, retain_graph=False)[0]
                if self.config.normalize_guidance:
                    norm = gradient.flatten(1).abs().mean(dim=1).clamp_min(1e-8)
                    gradient = gradient / norm[:, None, None, None]
                proposal = self.ddim_step(
                    residual.detach(),
                    epsilon.detach(),
                    timestep,
                    previous,
                    self.config.ddim_eta,
                )
                residual = proposal - scale * gradient.detach()
            else:
                with torch.no_grad():
                    epsilon = self.denoiser(residual, time_batch, rgb_latent)
                    residual = self.ddim_step(
                        residual,
                        epsilon,
                        timestep,
                        previous,
                        self.config.ddim_eta,
                    )
        return residual.detach()

    def reconstruct(
        self,
        rgb: torch.Tensor,
        sampling_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        rgb_padded, padding = _pad_to_multiple(rgb, self.required_spatial_multiple)
        with torch.no_grad():
            rgb_latent = self.rgb_initializer(rgb_padded)
        residual = self.sample_residual(
            rgb_latent,
            observed_rgb=rgb_padded,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            initial_noise=initial_noise,
            generator=generator,
        )
        with torch.no_grad():
            latent = self.denormalize_latent(rgb_latent + residual)
            hsi = self.autoencoder.decode(latent)
            if self.config.clip_denoised:
                hsi = hsi.clamp(self.config.hsi_min, self.config.hsi_max)
        return _remove_padding(hsi, padding)

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        hsi: Optional[torch.Tensor] = None,
        stage: int = 3,
    ):
        if stage == 1:
            if hsi is None:
                raise ValueError("hsi is required for stage 1")
            return self.stage1(hsi)
        if stage == 2:
            if rgb is None:
                raise ValueError("rgb is required for stage 2")
            return self.stage2(rgb)
        if stage == 3:
            if rgb is None:
                raise ValueError("rgb is required for stage 3")
            return self.reconstruct(rgb)
        raise ValueError("stage must be 1, 2, or 3")


# Compatibility alias for concise imports.
ResidualLatentDiffusionRGB2HSI = LatentDiffusionRGB2HSI
