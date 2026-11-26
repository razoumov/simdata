import os
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from fnoModule import FNO2d

# ==========================================
# Relative L2 Loss; standard for FNO
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

dir = "/home/user01/simdata/data"
initialConditions = sorted(glob(os.path.join(dir, "training", "r*s0000.png")))
runs = [os.path.basename(ic)[1:5] for ic in initialConditions]
timeSteps = sorted(glob(os.path.join(dir, "training", "r0001s*.png")))
steps = [os.path.basename(ts)[6:10] for ts in timeSteps]

nruns, nsteps = len(runs), len(steps)
nx, ny = np.asarray(Image.open(initialConditions[0]).convert("L"), dtype=np.float32).shape   # image spatial dimensions

xdata = np.zeros((nruns, 1, nx, ny), dtype=np.float32)        # initial conditions
ydata = np.zeros((nruns, nsteps, nx, ny), dtype=np.float32)   # 50 timesteps

for r, run in enumerate(tqdm(runs, desc="Loading training data")):
    initial = os.path.join(dir, "training", f"r{run}s0000.png")
    xdata[r, 0] = np.asarray(Image.open(initial).convert("L"), dtype=np.float32) / 255.0
    for s, step in enumerate(steps):
        timestep = os.path.join(dir, "training", f"r{run}s{step}.png")
        ydata[r, s] = np.asarray(Image.open(timestep).convert("L"), dtype=np.float32) / 255.0

xdata, ydata = torch.from_numpy(xdata), torch.from_numpy(ydata)

# ==========================================
# Train
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

checkpointFile = os.path.join(dir, "a04.pt")
torch.save(model.state_dict(), checkpointFile)
print("Saved model weights to", checkpointFile)
