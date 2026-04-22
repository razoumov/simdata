import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------
# 1. Spectral Convolution Layer (The core of FNO)
# ---------------------------------------------------------
class SpectralConv2d(nnx.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # number of lowest Fourier x-frequencies to keep; higher frequencies (fine noise) are discarded
        self.modes2 = modes2 # number of lowest Fourier y-frequencies to keep; higher frequencies (fine noise) are discarded
        scale = 1 / (in_channels * out_channels) # normalization to keep the signal variance stable during initialization
        # learnable weights are complex-valued in the Fourier domain
        self.weights1 = nnx.Param(
            scale * jax.random.normal(rngs.params(), (in_channels, out_channels, modes1, modes2), dtype=jnp.complex64)
        )
        self.weights2 = nnx.Param(
            scale * jax.random.normal(rngs.params(), (in_channels, out_channels, modes1, modes2), dtype=jnp.complex64)
        )
    def __call__(self, x):
        # x shape: (batch, height, width, channels)
        batch, h, w, c = x.shape
        # transform to Fourier domain
        x = jnp.transpose(x, (0, 3, 1, 2)) # reorder to (batch, channel, height, width) dimensions
        x_ft = jnp.fft.rfftn(x, axes=(-2, -1)) # Real Fast Fourier Transform: image pixels -> frequency coefficients
        # Multiply relevant Fourier modes
        out_ft = jnp.zeros((batch, self.out_channels, h, w // 2 + 1), dtype=jnp.complex64) # creates a blank canvas
                                                                               # in the frequency domain
        # top left corner
        res1 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1) # matrix
                                    # multiplication between the input frequencies and the learnable weights
                                    # across the input (i) and output (o) channels
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(res1)
        # bottom left corner
        res2 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(res2)
        # as for the top/bottom right corners, they are redundant, so no need to store them
        # return to physical space
        x = jnp.fft.irfftn(out_ft, s=(h, w), axes=(-2, -1)) # the Inverse FFT: filtered coefficients -> 2D image
        return jnp.transpose(x, (0, 2, 3, 1)) # reorder to (batch, height, width, channel) dimensions

# ---------------------------------------------------------
# 2. Full FNO Model
# ---------------------------------------------------------
class FNO2d(nnx.Module):   # high-level architecture to stack several Spectral Convolutions layers
    def __init__(self, modes, width, in_channels, out_channels, rngs: nnx.Rngs):
        # lift the input data into a higher-dim space (width) so the model has more "room" to learn complex features
        self.fc0 = nnx.Linear(in_channels, width, rngs=rngs)
        # four spectral convolution layers
        self.conv0 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv1 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv2 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv3 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        # skip connections: standard linear layers applied in the spatial domain
        self.w0 = nnx.Linear(width, width, rngs=rngs)
        self.w1 = nnx.Linear(width, width, rngs=rngs)
        self.w2 = nnx.Linear(width, width, rngs=rngs)
        self.w3 = nnx.Linear(width, width, rngs=rngs)
        # projection layers: squeeze high-dim space back down to the desired output size
        self.fc1 = nnx.Linear(width, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, out_channels, rngs=rngs)
    def __call__(self, x):
        # x: (batch, h, w, in_channels)
        x = self.fc0(x)
        # FNO iterations
        x1 = self.conv0(x) + self.w0(x) # add global patterns (from Fourier domain) and local patterns (from linear layer)
        x = jax.nn.gelu(x1)             # non-linear activation function
        x2 = self.conv1(x) + self.w1(x)
        x = jax.nn.gelu(x2)
        x3 = self.conv2(x) + self.w2(x)
        x = jax.nn.gelu(x3)
        x4 = self.conv3(x) + self.w3(x)
        # final projection
        x = self.fc1(x4)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x
