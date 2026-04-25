import jax
import jax.numpy as jnp
from flax import nnx

# have not tested this yet

# Key Improvements Made:Modular Design: Used a ConvBlock. Real U-Nets repeat the "Conv-BN-ReLU" pattern twice at
# every stage.Batch Normalization: Added nnx.BatchNorm. Without this, training a deep U-Net is incredibly
# difficult because gradients can vanish or explode.Max Pooling: Switched from strided convolutions to
# max_pool. While strided convs are fine, classic U-Net uses Max Pooling for downsampling.1x1 Convolution Head:
# The final output uses a $1 \times 1$ kernel. This maps the high-dimensional feature map down to your desired
# number of classes (e.g., 1 for binary segmentation) without blending spatial information further.Symmetry:
# Notice the filter counts: $64 \to 128 \to 256 \text{ (bottleneck) } \to 128 \to 64$. This balance is what
# gives the U-Net its name.

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
        # Encoder (Downsampling)
        self.enc1 = ConvBlock(1, 64, rngs=rngs)
        self.pool1 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.enc2 = ConvBlock(64, 128, rngs=rngs)
        self.pool2 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Bottleneck
        self.bottleneck = ConvBlock(128, 256, rngs=rngs)
        # Decoder (Upsampling)
        self.up2 = nnx.ConvTranspose(256, 128, kernel_size=(2, 2), strides=2, rngs=rngs)
        self.dec2 = ConvBlock(256, 128, rngs=rngs) # 128 (up) + 128 (skip) = 256
        self.up1 = nnx.ConvTranspose(128, 64, kernel_size=(2, 2), strides=2, rngs=rngs)
        self.dec1 = ConvBlock(128, 64, rngs=rngs) # 64 (up) + 64 (skip) = 128
        # Final Head
        self.final = nnx.Conv(64, 1, kernel_size=(1, 1), rngs=rngs)
    def __call__(self, x):
        # Encoder
        s1 = self.enc1(x)       # Skip connection 1
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)      # Skip connection 2
        p2 = self.pool2(s2)
        # Bottleneck
        b = self.bottleneck(p2)
        # Decoder
        u2 = self.up2(b)
        # Concatenate skip connection s2
        u2 = jnp.concatenate([u2, s2], axis=-1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        # Concatenate skip connection s1
        u1 = jnp.concatenate([u1, s1], axis=-1)
        d1 = self.dec1(u1)        
        return self.final(d1)
