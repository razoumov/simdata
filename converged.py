import jax
import jax.numpy as jnp
from flax import nnx
import optax
from PIL import Image
import glob
import numpy as np
import orbax.checkpoint as ocp

# ---------------------------------------------------------
# Load the image pairs for training.
# ---------------------------------------------------------

# find all initial condition files matching frame8****0000.png
dir = "/scratch/razoumov/jax/"
inputFiles = sorted(glob.glob(dir+"data/training/frame8*0000.png"))
print(f"Found {len(inputFiles)} initial conditions to process.")

# create two lists: initial conditions and solutions
X_list, Y_list = [], []
for i, initial in enumerate(inputFiles):
    runNumber = initial[-12:-8]
    if (i+1)%100 == 0:
        print(f"Reading run {runNumber}")
    solution = dir + "data/training/" + f"frame8{runNumber}0001.png"        
    img_x = Image.open(initial).convert('L')   # open image in grayscale (L) mode
    x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
    img_y = Image.open(solution).convert('L')            
    y_array = np.asarray(img_y, dtype=np.float32) / 255.0
    X_list.append(x_array)
    Y_list.append(y_array)

# convert these lists to JAX arrays and add the channel dimension (C=1 for grayscale)
X = jnp.array(X_list)[..., jnp.newaxis]
Y = jnp.array(Y_list)[..., jnp.newaxis]
print(f"Data shapes: X={X.shape}, Y={Y.shape}")

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
        res1 = jnp.einsum("bixy,ioxy->boxy", 
                          x_ft[:, :, :self.modes1, :self.modes2], 
                          self.weights1)
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].set(res1)
        # Lower corner
        res2 = jnp.einsum("bixy,ioxy->boxy", 
                          x_ft[:, :, -self.modes1:, :self.modes2], 
                          self.weights2)
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].set(res2)
        # Return to physical space
        x = jnp.fft.irfftn(out_ft, s=(h, w), axes=(-2, -1))
        return jnp.transpose(x, (0, 2, 3, 1)) # (B, H, W, C)

# ---------------------------------------------------------
# 2. Full FNO Model
# ---------------------------------------------------------
class FNO2d(nnx.Module):
    def __init__(self, modes, width, rngs: nnx.Rngs):
        # Lifting Layer
        self.fc0 = nnx.Linear(1, width, rngs=rngs)
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
        self.fc2 = nnx.Linear(128, 1, rngs=rngs)
    def __call__(self, x):
        # x: (batch, h, w, 1)
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

# Hyperparameters
modes = 12
width = 32
learning_rate = 1e-3
rngs = nnx.Rngs(params=0)

# Initialize Model & Optimizer
model = FNO2d(modes, width, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, batch_x, batch_y):
    def loss_fn(model):
        preds = model(batch_x)
        return jnp.mean((preds - batch_y)**2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# ---------------------------------------------------------
# The Training Loop. To keep it GPU-efficient, we slice our JAX arrays into mini-batches.
# ---------------------------------------------------------

numEpochs = 500  # 500 default for a less noisy solution
batchSize = 8    # default 8
numSamples = X.shape[0]

for epoch in range(numEpochs):
    perms = jax.random.permutation(jax.random.PRNGKey(epoch), numSamples)
    X_shuffled, Y_shuffled = X[perms], Y[perms] # to be used instead of X,Y below
    epoch_loss = []
    for i in range(0, numSamples, batchSize):
        bx = X_shuffled[i : i + batchSize]
        by = Y_shuffled[i : i + batchSize]
        loss = train_step(model, optimizer, bx, by)
        epoch_loss.append(loss)
    print(f"Epoch {epoch}, loss: {np.mean(epoch_loss):.6f}")
    graph, state = nnx.split(model)   # extract the state from NNX
    checkpointer = ocp.StandardCheckpointer()
    if (epoch+1)%10 == 0:
        print('/scratch/razoumov/jax/weights%03d'%(epoch))
        checkpointer.save('/scratch/razoumov/jax/weights%03d'%(epoch), state)

checkpointer.wait_until_finished()   # wait for the save thread to finish writing to disk
