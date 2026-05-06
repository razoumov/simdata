import jax
import jax.numpy as jnp
from flax import nnx

# 1. add ConvBlock to repeat the "Conv-BN-ReLU" pattern twice at every stage
# 2. add nnx.BatchNorm to make sure that gradients do not vanish or explode
# 3. switch from strided convolutions to max_pool for downsampling, to reflect the classic U-Net
# 4. uses a 1x1 kernel for the final output to map the high-dimensional feature map down to a single variable
# 5. symmetry: filter symmetry: 1 -> 64 -> 128 -> 256 -> 128 -> 64 -> 1, to reflect the classic U-Net

class ConvBlock(nnx.Module):
    """The standard U-Net double-convolution block."""
    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)
        self.conv2 = nnx.Conv(out_features, out_features, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)
    def __call__(self, x):
        x = nnx.relu(self.bn1(self.conv1(x)))
        x = nnx.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # encoder (downsampling)
        self.enc1 = ConvBlock(1, 64, rngs=rngs)
        self.pool1 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.enc2 = ConvBlock(64, 128, rngs=rngs)
        self.pool2 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.bottleneck = ConvBlock(128, 256, rngs=rngs)
        # decoder (upsampling)
        self.up2 = nnx.ConvTranspose(256, 128, kernel_size=(2, 2), strides=2, rngs=rngs)
        self.dec2 = ConvBlock(256, 128, rngs=rngs) # 128 (up) + 128 (skip) = 256
        self.up1 = nnx.ConvTranspose(128, 64, kernel_size=(2, 2), strides=2, rngs=rngs)
        self.dec1 = ConvBlock(128, 64, rngs=rngs) # 64 (up) + 64 (skip) = 128
        self.final = nnx.Conv(64, 1, kernel_size=(1, 1), rngs=rngs)
    def __call__(self, x):
        s1 = self.enc1(x)         # skip connection 1
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)        # skip connection 2
        p2 = self.pool2(s2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        u2 = jnp.concatenate([u2, s2], axis=-1)   # concatenate skip connection s2
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = jnp.concatenate([u1, s1], axis=-1)   # concatenate skip connection s1
        d1 = self.dec1(u1)        
        return self.final(d1)
