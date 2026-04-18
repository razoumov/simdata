import jax
import jax.numpy as jnp
from flax import nnx
import optax
import glob
import numpy as np
from PIL import Image
import orbax.checkpoint as ocp

modelName = 3

# Load the image pairs (In1​,Out1​) into NumPy arrays X_train and Y_train:

# find all initial condition files matching frame8****0000.png
dir = "/scratch/razoumov/jax/"
inputFiles = sorted(glob.glob(dir+"data/training/frame8*0000.png"))
print(f"Found {len(inputFiles)} initial conditions to process.")

# create two lists: initial conditions and solutions
X_list, Y_list = [], []
for initial in inputFiles:
    runNumber = initial[-12:-8]
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

# Define the model

match modelName:
    case 1:
        from unetSimple import UNet
    case 2:
        from unetComplex import UNet, UNetBlock
    case 3:
        from fourier import SpectralConv2d, FNO2d

# initialize model and optimizer
match modelName:
    case 2:
        rngs = nnx.Rngs(0)
        model = UNet(in_features=1, out_features=1, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    case 3:
        modes, width, learning_rate = 12, 32, 1e-3
        rngs = nnx.Rngs(params=0)
        model = FNO2d(modes, width, in_channels=1, out_channels=1, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

# define the training function

@nnx.jit
def train_step(model, optimizer, batch_x, batch_y):
    def loss_fn(model):
        y_pred = model(batch_x)
        return jnp.mean((y_pred - batch_y)**2)   # mean squared error for residuals
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# ---------------------------------------------------------
# The Training Loop. To keep it GPU-efficient, we slice our JAX arrays into mini-batches.
# ---------------------------------------------------------

numEpochs = 50
batchSize = 8
numSamples = X.shape[0]

for epoch in range(numEpochs):
    perms = jax.random.permutation(jax.random.PRNGKey(epoch), numSamples)
    X_shuffled, Y_shuffled = X[perms], Y[perms]
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
        checkpointer.save('/scratch/razoumov/jax/weights%03d'%(epoch), state)








# Read from checkpoint

# dirname="weights009"
# dir = "/scratch/razoumov/jax/"
# checkpointer = ocp.StandardCheckpointer()
# graph, state = nnx.split(model)
# restored_state = checkpointer.restore(dir+dirname, target=state)   # restore the state
# nnx.update(model, restored_state)   # load back into the model






# # Infer from memory

# @nnx.jit
# def predict(model, x):
#     return model(x)

# img_x = Image.open(dir+'data/testing/frame800030000.png').convert('L')   # open image in grayscale (L) mode
# x_array = np.asarray(img_x, dtype=np.float32) / 255.0   # convert to NumPy array and normalize assuming 8-bit images
# initialState = jnp.array(x_array)[np.newaxis, ..., np.newaxis]
# # initialState = jnp.ones((1, 500, 500, 1))
# predictedSolution = predict(model, initialState).reshape(500, 500)

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(5, 5), dpi=100)
# ax = fig.add_axes([0, 0, 1, 1])
# ax.imshow(predictedSolution, interpolation='nearest', cmap='viridis')
# ax.axis('off')
# plt.savefig(dir + "prediction.png", pad_inches=0)
# plt.close()
