import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# ==========================================
# 1. Data Loading
# ==========================================
def load_simulation_data(data_dir: str, num_steps: int = 50):
    ic_files = sorted(glob(os.path.join(data_dir, "frame8*0000.png")))
    if not ic_files:
        raise FileNotFoundError(f"No initial condition files found in {data_dir}")

    runs = [os.path.basename(p)[6:10] for p in ic_files]
    steps = [f"{s:04d}" for s in range(1, num_steps + 1)]

    num_runs = len(runs)
    # Infer spatial dimensions from the first image
    sample_img = np.asarray(Image.open(ic_files[0]).convert("L"), dtype=np.float32) / 255.0
    H, W = sample_img.shape

    x_data = np.zeros((num_runs, 1, H, W), dtype=np.float32)
    y_data = np.zeros((num_runs, num_steps, H, W), dtype=np.float32)

    for r, run in enumerate(tqdm(runs, desc="Loading runs")):
        ic_path = os.path.join(data_dir, f"frame8{run}0000.png")
        x_data[r, 0] = np.asarray(Image.open(ic_path).convert("L"), dtype=np.float32) / 255.0

        for s, step in enumerate(steps):
            step_path = os.path.join(data_dir, f"frame8{run}{step}.png")
            y_data[r, s] = np.asarray(Image.open(step_path).convert("L"), dtype=np.float32) / 255.0

    return torch.from_numpy(x_data), torch.from_numpy(y_data)


# ==========================================
# 2. FNO2d Architecture
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
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(input_fft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_c, m1, m2) x (in_c, out_c, m1, m2) -> (batch, out_c, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", input_fft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape

        # Compute real 2D FFT: shape (B, C, H, W // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Clamp modes safely in case resolution is smaller than modes
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft = torch.zeros(
            batch_size, self.out_channels, height, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )

        # Positive and negative frequencies along dim -2 (height)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    """
    Maps initial condition u0 (1 channel + 2 spatial coordinates = 3 input channels)
    directly to all T future timesteps (T output channels).
    """

    def __init__(
            self,
            out_steps: int = 50,
            modes1: int = 24,
            modes2: int = 24,
            width: int = 32,
            num_layers: int = 4,
    ):
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
        return torch.cat((grid_x, grid_y), dim=1)  # (1, 2, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        batch_size, _, height, width = x.shape
        grid = self.get_grid((height, width), x.device).expand(batch_size, -1, -1, -1)

        # Concatenate initial state and spatial coordinates -> (B, H, W, 3)
        h = torch.cat((x, grid), dim=1).permute(0, 2, 3, 1)
        h = self.lift(h).permute(0, 3, 1, 2)  # (B, width, H, W)

        for idx, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            h = conv(h) + w(h)
            if idx < len(self.conv_layers) - 1:
                h = F.gelu(h)

        # Project channel dimension to all 50 timesteps -> (B, 50, H, W)
        out = h.permute(0, 2, 3, 1)
        out = F.gelu(self.proj1(out))
        out = self.proj2(out).permute(0, 3, 1, 2)
        return out


# ==========================================
# 3. Relative L2 Loss (Standard for FNO)
# ==========================================
class RelativeL2Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = pred.shape[0]
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1) + 1e-8
        return torch.mean(diff_norm / target_norm)








        # train_dir: str = "/home/user01/simdata/data/training/",
        # test_image_path: str = "test.png",
        # checkpoint_path: str = "fno2d_50steps.pt",
        # num_steps: int = 50,
        # batch_size: int = 4,
        # epochs: int = 100,
        # lr: float = 1e-3,
        # modes: int = 24,
        # width: int = 32,
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x_data, y_data = load_simulation_data(train_dir, num_steps=num_steps)
dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = FNO2d(out_steps=num_steps, modes1=modes, modes2=modes, width=width).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = RelativeL2Loss()

model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bx.size(0)

    scheduler.step()
    avg_loss = total_loss / len(dataset)
    if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
        print(f"Epoch [{epoch:03d}/{epochs:03d}] - Rel L2 Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), checkpoint_path)
print(f"Saved model weights to {checkpoint_path}")

# Inference on test.png -> predicts all 50 steps at once
model.eval()
test_img = np.asarray(Image.open(test_image_path).convert("L"), dtype=np.float32) / 255.0
test_tensor = torch.from_numpy(test_img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

with torch.no_grad():
    pred_steps = model(test_tensor).squeeze(0).cpu().numpy()  # (50, H, W)

predicted_50_steps = pred_steps



print("Predicted output shape:", predicted_50_steps.shape)  # (50, 500, 500)




# saving predicted frames
os.makedirs("test_predictions", exist_ok=True)
for idx, frame in enumerate(predicted_50_steps, start=1):
    frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(frame_u8, mode="L").save(f"test_predictions/pred_{idx:04d}.png")
