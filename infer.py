import jax
import optax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
from flax import linen as nn
from typing import Sequence
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# # define ConvBlock, same as before
# class ConvBlock(nn.Module):
#     features: int
#     kernel_size: Sequence[int] = (3, 3)
#     @nn.compact
#     def __call__(self, x, training: bool):
#         # 1. Convolution
#         x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='SAME')(x)
#         # 2. Batch Normalization (crucial for stability)
#         x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
#         # 3. Activation
#         x = nn.relu(x)
#         return x

# # define UNet, same as before
# class UNet(nn.Module):
#     @nn.compact
#     def __call__(self, x, training: bool):
#         c1 = ConvBlock(features=32)(x, training=training)
#         d2 = nn.max_pool(c1, window_shape=(2, 2), strides=(2, 2))
#         c2 = ConvBlock(features=64)(d2, training=training)
#         c_bottleneck = ConvBlock(features=128)(c2, training=training)
#         target_shape = c1.shape[1:3]  # Should be (501, 501)
#         u1 = jax.image.resize(c_bottleneck,
#                               shape=(c_bottleneck.shape[0], target_shape[0], target_shape[1], 64), method='nearest')
#         u1 = nn.Conv(features=64, kernel_size=(2, 2), padding='SAME')(u1)
#         u1 = nn.relu(u1) # u1 shape: (B, 501, 501, 64)
#         u1 = jnp.concatenate([u1, c1], axis=-1)
#         c_up1 = ConvBlock(features=64)(u1, training=training)
#         output = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME')(c_up1)
#         return output

import flax.linen as nn

# Residual Conv Block
class ResBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, training: bool):
        residual = x
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        # Project residual if channels differ
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1))(residual)
        x = x + residual
        x = nn.relu(x)
        return x

# Downsampling Block
class DownBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, training: bool):
        x = ResBlock(self.features)(x, training)
        skip = x
        x = nn.Conv(self.features, (3, 3), strides=(2, 2), padding="SAME")(x)
        return x, skip

# Upsampling Block
class UpBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, skip, training: bool):
        x = nn.ConvTranspose(
            self.features,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME"
        )(x)
        # Handle odd sizes (e.g., 501)
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = jax.image.resize(
                x,
                (x.shape[0], skip.shape[1], skip.shape[2], x.shape[3]),
                method="bilinear",
            )
        x = jnp.concatenate([x, skip], axis=-1)
        x = ResBlock(self.features)(x, training)
        return x

# Full U-Net
class UNet(nn.Module):
    base_features: int = 32
    out_channels: int = 1
    @nn.compact
    def __call__(self, x, training: bool):
        # Encoder
        x1, skip1 = DownBlock(self.base_features)(x, training)
        x2, skip2 = DownBlock(self.base_features * 2)(x1, training)
        x3, skip3 = DownBlock(self.base_features * 4)(x2, training)
        # Bottleneck
        bottleneck = ResBlock(self.base_features * 8)(x3, training)
        # Decoder
        u3 = UpBlock(self.base_features * 4)(bottleneck, skip3, training)
        u2 = UpBlock(self.base_features * 2)(u3, skip2, training)
        u1 = UpBlock(self.base_features)(u2, skip1, training)
        # Output layer
        output = nn.Conv(self.out_channels, (1, 1), padding="SAME")(u1)
        return output

@struct.dataclass
class TrainState(train_state.TrainState):
    batch_stats: dict

if len(sys.argv) < 2:
    print("Usage: python script.py weights.pkl")
    sys.exit(1)

filename = sys.argv[1]
with open(filename, 'rb') as f:
    loaded_data = pickle.load(f)

key = jax.random.PRNGKey(42)   # needed for initialization, but its value doesn't matter here
model = UNet()
dummy_input = jnp.zeros((1, 501, 501, 1))
optimizer = optax.adam(learning_rate=1e-4)   # optimizer definition must match

init_variables = model.init({'params': key, 'dropout': key}, dummy_input, training=True)

loaded_state = TrainState.create(
    apply_fn=model.apply,
    params=loaded_data['params'],
    tx=optimizer,
    batch_stats=loaded_data['batch_stats'],
)

dir = "./"
initial = dir + "data/testing/frame800030000.png"
img_x = Image.open(initial).convert('L')   # open image in grayscale (L) mode
x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
initialState = x_array.reshape(1, 501, 501, 1)

@jax.jit
def predict_step(state, inputs):
    """Performs a single inference step."""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    predictions = UNet().apply(variables, inputs, training=False)
    return predictions

prediction = predict_step(loaded_state, initialState).reshape(501, 501)

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(prediction, interpolation='nearest', cmap='viridis')
plt.savefig(dir+"prediction.png")
plt.close(fig)
