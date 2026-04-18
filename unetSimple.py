# Define the model (NNX). We will build a simplified U-Net. NNX allows us to define models as Python classes
# that hold their own state, making the code much cleaner than traditional functional JAX.

class UNet(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        # Downsampling
        self.c1 = nnx.Conv(in_features, 32, kernel_size=(3, 3), rngs=rngs)
        self.c2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, rngs=rngs)
        # Bottleneck
        self.bottleneck = nnx.Conv(64, 64, kernel_size=(3, 3), rngs=rngs)
        # Upsampling
        self.up = nnx.ConvTranspose(64, 32, kernel_size=(3, 3), strides=2, rngs=rngs)
        self.out = nnx.Conv(32, out_features, kernel_size=(3, 3), rngs=rngs)
    def __call__(self, x):
        # Encoder
        x1 = nnx.relu(self.c1(x))
        x2 = nnx.relu(self.c2(x1))
        # Bottleneck
        x = nnx.relu(self.bottleneck(x2))
        # Decoder (Simplified skip connection logic)
        x = nnx.relu(self.up(x))
        # Ensure shapes match for a simple residual add or concatenation if needed
        return jax.nn.sigmoid(self.out(x)) # Sigmoid if data is normalized [0, 1]
