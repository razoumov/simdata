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

# ==========================================
# Relative L2 Loss (Standard for FNO)
# ==========================================
class RelativeL2Loss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = pred.shape[0]
        pred_flat = pred.reshape(batch_size, -1)
        target_flat = target.reshape(batch_size, -1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1) + 1e-8
        return torch.mean(diff_norm / target_norm)










# ==========================================
# Load data
# ==========================================

dir = "/home/user01/simdata/data/training"
initialConditions = sorted(glob(os.path.join(dir, "frame8*0000.png")))
runs = [os.path.basename(ic)[6:10] for ic in initialConditions]
timeSteps = sorted(glob(os.path.join(dir, "frame80001*.png")))
steps = [os.path.basename(ts)[10:14] for ts in timeSteps]

nruns, nsteps = len(runs), len(steps)
nx, ny = np.asarray(Image.open(initialConditions[0]).convert("L"), dtype=np.float32).shape   # image spatial dimensions

xdata = np.zeros((nruns, 1, nx, ny), dtype=np.float32)        # initial conditions
ydata = np.zeros((nruns, nsteps, nx, ny), dtype=np.float32)   # 50 timesteps

for r, run in enumerate(tqdm(runs, desc="Loading training data")):
    initial = os.path.join(dir, f"frame8{run}0000.png")
    xdata[r, 0] = np.asarray(Image.open(initial).convert("L"), dtype=np.float32) / 255.0
    for s, step in enumerate(steps):
        timestep = os.path.join(dir, f"frame8{run}{step}.png")
        ydata[r, s] = np.asarray(Image.open(timestep).convert("L"), dtype=np.float32) / 255.0

xdata, ydata = torch.from_numpy(xdata), torch.from_numpy(ydata)






# ==========================================
# Training & Inference on test.png
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 4
epochs = 100
lr = 1e-3
modes = 24
width = 32

dataset = TensorDataset(xdata, ydata)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = FNO2d(out_steps=nsteps, modes1=modes, modes2=modes, width=width).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = RelativeL2Loss()

model.train()   # put nn.Module into training mode
for epoch in range(1, epochs + 1):
    print(epoch)
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

checkpointFile = "/home/user01/simdata/data/a01.pt"
torch.save(model.state_dict(), checkpointFile)
print("Saved model weights to", checkpointFile

# Inference on a test image to predict all 50 steps at once
model.eval()   # put nn.Module into evaluation/inference mode
test_num = 5
testImageFile = f"/home/user01/simdata/data/testing/frame8{test_num:04d}0000.png"
testImage = np.asarray(Image.open(testImageFile).convert("L"), dtype=np.float32) / 255.0
testTensor = torch.from_numpy(testImage).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, nx, ny)

with torch.no_grad():
    pred_steps = model(testTensor).squeeze(0).cpu().numpy()  # (50, nx, ny)

# print("Predicted output shape:", pred_steps.shape)  # (50, 500, 500)

# # saving initial+50 predicted frames
# for idx, frame in enumerate(pred_steps, start=1):
#     frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
#     Image.fromarray(frame_u8, mode="L").save(f"/home/user01/simdata/data/testing/pred_{idx:04d}.png")

# save the last predicted frame
idx = 50
frame = np.clip(pred_steps[idx] * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(frame, mode="L").save(f"/home/user01/simdata/data/testing/pred{test_num:1d}.png")
