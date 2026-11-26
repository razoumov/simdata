import os
import numpy as np
from PIL import Image
import torch
from fnoModule import FNO2d

dir = "/home/user01/simdata/data"
checkpointFile = os.path.join(dir, "a04.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Recreate the model with the same hyperparameters used when training; predict all 50 steps at once
modes = 24
width = 32
nsteps = 1
model = FNO2d(out_steps=1+nsteps, modes1=modes, modes2=modes, width=width).to(device) # num_layers=4

# Load the saved weights (map_location handles CPU/GPU differences safely)
state_dict = torch.load(checkpointFile, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.eval()   # put nn.Module into evaluation/inference mode
case = "line"
for test_num in range(1,6):
    testImageFile = os.path.join(dir, case, f"r{test_num:04d}s0000.png")
    testImage = np.asarray(Image.open(testImageFile).convert("L"), dtype=np.float32) / 255.0
    testTensor = torch.from_numpy(testImage).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, nx, ny)
    with torch.no_grad():
        pred_steps = model(testTensor).squeeze(0).cpu().numpy()
    print("Predicted output shape:", pred_steps.shape)  # (51, 500, 500)
    # # save initial + all predicted frames
    # for idx, frame in enumerate(pred_steps, start=1):
    #     frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    #     Image.fromarray(frame_u8, mode="L").save(os.path.join(dir, "testing", f"pred{idx:04d}.png"))
    # save the last predicted frame
    idx = nsteps
    frame = np.clip(pred_steps[idx] * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(frame, mode="L").save(os.path.join(dir, case, f"pred{test_num:1d}.png"))
