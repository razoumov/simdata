import jax
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from PIL import Image
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from caeSimple import CAE
rngs = nnx.Rngs(0)   # to provide random keys to initialize the parameters
model = CAE(rngs=rngs)

if len(sys.argv) < 3:
    print("Usage: python script.py weightsDir initialImage.png")
    sys.exit(1)

dirname = sys.argv[1]
currentPath = str(Path('.').resolve()) + '/'
checkpointer = ocp.StandardCheckpointer()
graph, state = nnx.split(model)
restored_state = checkpointer.restore(currentPath+dirname, target=state)   # restore the state
nnx.update(model, restored_state)   # load back into the model

@nnx.jit
def predict(model, x):
    return model(x)

img_x = Image.open(sys.argv[2]).convert('L')   # open image in grayscale (L) mode
x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
initialState = jnp.array(x_array)[np.newaxis, ..., np.newaxis]
predictedSolution = predict(model, initialState).reshape(500, 500)

fig = plt.figure(figsize=(5, 5), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(predictedSolution, interpolation='nearest', cmap='viridis')
ax.axis('off')
plt.savefig("prediction.png", pad_inches=0)
plt.close()
