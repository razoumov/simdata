import jax
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from PIL import Image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys

# Define the model

# class UNet(nnx.Module):
#     def __init__(self, in_features, out_features, rngs: nnx.Rngs):
#         # Downsampling
#         self.c1 = nnx.Conv(in_features, 32, kernel_size=(3, 3), rngs=rngs)
#         self.c2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, rngs=rngs)
#         # Bottleneck
#         self.bottleneck = nnx.Conv(64, 64, kernel_size=(3, 3), rngs=rngs)
#         # Upsampling
#         self.up = nnx.ConvTranspose(64, 32, kernel_size=(3, 3), strides=2, rngs=rngs)
#         self.out = nnx.Conv(32, out_features, kernel_size=(3, 3), rngs=rngs)
#     def __call__(self, x):
#         # Encoder
#         x1 = nnx.relu(self.c1(x))
#         x2 = nnx.relu(self.c2(x1))
#         # Bottleneck
#         x = nnx.relu(self.bottleneck(x2))
#         # Decoder (Simplified skip connection logic)
#         x = nnx.relu(self.up(x))
#         # Ensure shapes match for a simple residual add or concatenation if needed
#         return jax.nn.sigmoid(self.out(x)) # Sigmoid if data is normalized [0, 1]

class UNetBlock(nnx.Module):
    def __init__(self, in_chan, out_chan, rngs: nnx.Rngs, stride=1):
        self.conv = nnx.Conv(in_chan, out_chan, kernel_size=(3, 3), strides=stride, padding='SAME', rngs=rngs)
        self.bn = nnx.BatchNorm(out_chan, momentum=0.9, rngs=rngs)
    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nnx.relu(x)

class UNet(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        # Encoder: 32 -> 64 -> 128 -> 256
        self.enc1 = UNetBlock(in_features, 32, rngs)
        self.enc2 = UNetBlock(32, 64, rngs, stride=2)
        self.enc3 = UNetBlock(64, 128, rngs, stride=2)
        self.enc4 = UNetBlock(128, 256, rngs, stride=2)
        # Bottleneck: capped at 512
        self.bottleneck = UNetBlock(256, 512, rngs)
        # Decoder
        self.up4 = nnx.ConvTranspose(512, 256, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec4 = UNetBlock(512, 256, rngs) # (256 up + 256 skip)
        self.up3 = nnx.ConvTranspose(256, 128, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec3 = UNetBlock(256, 128, rngs) # (128 up + 128 skip)
        self.up2 = nnx.ConvTranspose(128, 64, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec2 = UNetBlock(128, 64, rngs)  # (64 up + 64 skip)
        self.final_conv = nnx.Conv(64, out_features, kernel_size=(1, 1), rngs=rngs)
    def __call__(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        # Helper to handle the shape matching for skip connections
        def upsample_and_concat(current, skip):
            # current.shape[0] is batch, skip.shape[1:3] is HW, current.shape[-1] is C
            target_shape = (current.shape[0], skip.shape[1], skip.shape[2], current.shape[3])
            up = jax.image.resize(current, target_shape, method="bilinear")
            return jnp.concatenate([up, skip], axis=-1)
        # Decoder path
        x = upsample_and_concat(self.up4(b), s4)
        x = self.dec4(x)
        x = upsample_and_concat(self.up3(x), s3)
        x = self.dec3(x)
        x = upsample_and_concat(self.up2(x), s2)
        x = self.dec2(x)
        # Final resize to exactly 500x500 before the last conv
        final_shape = (x.shape[0], 500, 500, x.shape[3])
        x = jax.image.resize(x, final_shape, method="bilinear")        
        return jax.nn.sigmoid(self.final_conv(x))

rngs = nnx.Rngs(0)
model = UNet(in_features=1, out_features=1, rngs=rngs)

# Read the checkpoint

if len(sys.argv) < 2:
    print("Usage: python script.py weightsDir")
    sys.exit(1)

dirname = sys.argv[1]
dir = "/scratch/razoumov/jax/"
checkpointer = ocp.StandardCheckpointer()
graph, state = nnx.split(model)
restored_state = checkpointer.restore(dir+dirname, target=state)   # restore the state
nnx.update(model, restored_state)   # load back into the model

# Infer

@nnx.jit
def predict(model, x):
    return model(x)

img_x = Image.open(dir+'data/testing/frame800030000.png').convert('L')   # open image in grayscale (L) mode
x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
initialState = jnp.array(x_array)[np.newaxis, ..., np.newaxis]
# initialState = jnp.ones((1, 500, 500, 1))
predictedSolution = predict(model, initialState).reshape(500, 500)

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(predictedSolution, interpolation='nearest', cmap='viridis')
plt.savefig(dir+"prediction.png")
plt.close(fig)
