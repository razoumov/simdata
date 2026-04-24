import jax
import jax.numpy as jnp
from flax import nnx

# Define a simplified U-Net model with NNX. NNX allows us to define models as Python classes that hold their
# own state, making the code much cleaner than traditional functional JAX.

class UNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):   # define the model layers
        self.c1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs) # convolution layer applying a sliding kernel over input data
                  # arguments: number of input channels (1), number of output channels/filters (32),
                  # convolution window kernel size (3x3), dictionary of random seeds
        self.c2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, rngs=rngs) # downsampling layer to reduce the width/height by 2X
        self.bottleneck = nnx.Conv(64, 64, kernel_size=(3, 3), rngs=rngs) # bottleneck processing the most compressed data representation
        self.up = nnx.ConvTranspose(64, 32, kernel_size=(3, 3), strides=2, rngs=rngs) # upsampling layer to double the width/height by 2X
        self.out = nnx.Conv(32, 1, kernel_size=(3, 3), rngs=rngs) # final layer to map the features back to a single channel
    def __call__(self, x):   # define the forward pass
        x1 = nnx.relu(self.c1(x))   # the encoder (downsampling)
        x2 = nnx.relu(self.c2(x1))
        x3 = nnx.relu(self.bottleneck(x2))   # the bottleneck
        x4 = nnx.relu(self.up(x3))   # the decoder (upsampling)
        # x4 = jnp.concatenate([x4, x1], axis=-1)   # skip connection to the first layer
        return jax.nn.sigmoid(self.out(x4)) # back to 1 channel, sigmoid ensures all values are between 0 and 1
