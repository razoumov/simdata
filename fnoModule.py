import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# FNO2d Architecture
# ==========================================
class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT -> linear transform on low-frequency modes -> iFFT."""
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes along height (y)
        self.modes2 = modes2  # Number of Fourier modes along width (x)
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
    @staticmethod
    def compl_mul2d(input_fft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_c, m1, m2) x (in_c, out_c, m1, m2) -> (batch, out_c, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", input_fft, weights)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        # Compute real 2D FFT: shape (B, C, nx, ny // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        # Clamp modes safely in case resolution is smaller than modes
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size, self.out_channels, height, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        # Positive and negative frequencies along dim -2 (height)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")

class FNO2d(nn.Module):
    """
    Maps initial condition u0 (1 channel + 2 spatial coordinates = 3 input channels)
    directly to all T future timesteps (T output channels).
    """
    def __init__(self, out_steps: int = 50, modes1: int = 24, modes2: int = 24, width: int = 32, num_layers: int = 4):
        super().__init__()
        self.out_steps = out_steps
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        # +2 channels for normalized (x, y) spatial grid coordinates
        self.lift = nn.Linear(3, self.width)
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=1)
            for _ in range(num_layers)
        ])
        self.proj1 = nn.Linear(self.width, 128)
        self.proj2 = nn.Linear(128, self.out_steps)
    @staticmethod
    def get_grid(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        height, width = shape
        grid_y = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1).expand(1, 1, height, width)
        grid_x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width).expand(1, 1, height, width)
        return torch.cat((grid_x, grid_y), dim=1)  # (1, 2, nx, ny)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, nx, ny)
        batch_size, _, height, width = x.shape
        grid = self.get_grid((height, width), x.device).expand(batch_size, -1, -1, -1)
        # Concatenate initial state and spatial coordinates -> (B, nx, ny, 3)
        h = torch.cat((x, grid), dim=1).permute(0, 2, 3, 1)
        h = self.lift(h).permute(0, 3, 1, 2)  # (B, width, nx, ny)
        for idx, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            h = conv(h) + w(h)
            if idx < len(self.conv_layers) - 1:
                h = F.gelu(h)
        # Project channel dimension to all 50 timesteps -> (B, 50, nx, ny)
        out = h.permute(0, 2, 3, 1)
        out = F.gelu(self.proj1(out))
        out = self.proj2(out).permute(0, 3, 1, 2)
        return out
