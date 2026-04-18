# ---------------------------------------------------------
# 1. Spectral Convolution Layer (The core of FNO)
# ---------------------------------------------------------
class SpectralConv2d(nnx.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rngs: nnx.Rngs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to keep
        self.modes2 = modes2
        # Weights are complex-valued in the Fourier domain
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nnx.Param(
            scale * jax.random.normal(rngs.params(), (in_channels, out_channels, modes1, modes2), dtype=jnp.complex64)
        )
        self.weights2 = nnx.Param(
            scale * jax.random.normal(rngs.params(), (in_channels, out_channels, modes1, modes2), dtype=jnp.complex64)
        )
    def __call__(self, x):
        # x shape: (batch, height, width, channels)
        batch, h, w, c = x.shape
        # Transform to Fourier domain
        x = jnp.transpose(x, (0, 3, 1, 2)) # (B, C, H, W)
        x_ft = jnp.fft.rfftn(x, axes=(-2, -1))
        # Multiply relevant Fourier modes
        out_ft = jnp.zeros((batch, self.out_channels, h, w // 2 + 1), dtype=jnp.complex64)
        # Upper corner
        res1 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(res1)
        # Lower corner
        res2 = jnp.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(res2)
        # Return to physical space
        x = jnp.fft.irfftn(out_ft, s=(h, w), axes=(-2, -1))
        return jnp.transpose(x, (0, 2, 3, 1)) # (B, H, W, C)

# ---------------------------------------------------------
# 2. Full FNO Model
# ---------------------------------------------------------
class FNO2d(nnx.Module):
    def __init__(self, modes, width, in_channels, out_channels, rngs: nnx.Rngs):
        # Lifting Layer
        self.fc0 = nnx.Linear(in_channels, width, rngs=rngs)
        # Spectral Layers
        self.conv0 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv1 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv2 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        self.conv3 = SpectralConv2d(width, width, modes, modes, rngs=rngs)
        # Skip connections (W in FNO papers)
        self.w0 = nnx.Linear(width, width, rngs=rngs)
        self.w1 = nnx.Linear(width, width, rngs=rngs)
        self.w2 = nnx.Linear(width, width, rngs=rngs)
        self.w3 = nnx.Linear(width, width, rngs=rngs)
        # Projection Layers
        self.fc1 = nnx.Linear(width, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, out_channels, rngs=rngs)
    def __call__(self, x):
        # x: (batch, h, w, in_channels)
        x = self.fc0(x)
        # FNO Iterations
        x1 = self.conv0(x) + self.w0(x)
        x = jax.nn.gelu(x1)
        x2 = self.conv1(x) + self.w1(x)
        x = jax.nn.gelu(x2)
        x3 = self.conv2(x) + self.w2(x)
        x = jax.nn.gelu(x3)
        x4 = self.conv3(x) + self.w3(x)
        # Final Projection
        x = self.fc1(x4)
        x = jax.nn.gelu(x)
        x = self.fc2(x)
        return x
