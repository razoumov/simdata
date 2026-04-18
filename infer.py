import jax
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from PIL import Image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys

modelName = 3

# ---------------------------------------------------------
# Define the model.
# ---------------------------------------------------------

match modelName:
    case 1:
        from unetSimple import UNet
    case 2:
        from unetComplex import UNet, UNetBlock
        rngs = nnx.Rngs(0)
        model = UNet(in_features=1, out_features=1, rngs=rngs)
    case 3:
        from fourier import SpectralConv2d, FNO2d
        # Hyperparameters
        modes = 12
        width = 32
        rngs = nnx.Rngs(params=0)
        model = FNO2d(modes, width, in_channels=1, out_channels=1, rngs=rngs)

# ---------------------------------------------------------
# Read the checkpoint
# ---------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python script.py weightsDir")
    sys.exit(1)

dirname = sys.argv[1]
dir = "/scratch/razoumov/jax/"
checkpointer = ocp.StandardCheckpointer()
graph, state = nnx.split(model)
restored_state = checkpointer.restore(dir+dirname, target=state)   # restore the state
nnx.update(model, restored_state)   # load back into the model

# ---------------------------------------------------------
# Infer and plot.
# ---------------------------------------------------------

@nnx.jit
def predict(model, x):
    return model(x)

img_x = Image.open(dir+'data/testing/frame800030000.png').convert('L')   # open image in grayscale (L) mode
x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
initialState = jnp.array(x_array)[np.newaxis, ..., np.newaxis]
# initialState = jnp.ones((1, 500, 500, 1))
predictedSolution = predict(model, initialState).reshape(500, 500)

fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(predictedSolution, interpolation='nearest', cmap='viridis')
ax.axis('off')
plt.savefig(dir + "prediction.png", pad_inches=0)
plt.close()
