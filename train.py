import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from flax.training import train_state
import optax
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# # The U-Net consists of an Encoder (downsampling path) and a Decoder (upsampling path) with skip connections.

# class ConvBlock(nn.Module):
#     features: int
#     kernel_size: Sequence[int] = (3, 3)
#     @nn.compact
#     def __call__(self, x, training: bool):
#         # 1. Convolution
#         x = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='SAME')(x)
#         # 2. Batch Normalization (crucial for stability)
#         x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
#         # 3. Activation
#         x = nn.relu(x)
#         return x

# class UNet(nn.Module):
#     @nn.compact
#     def __call__(self, x, training: bool):
#         c1 = ConvBlock(features=32)(x, training=training)
#         d2 = nn.max_pool(c1, window_shape=(2, 2), strides=(2, 2))
#         c2 = ConvBlock(features=64)(d2, training=training)
#         c_bottleneck = ConvBlock(features=128)(c2, training=training)
#         target_shape = c1.shape[1:3]  # Should be (501, 501)
#         u1 = jax.image.resize(c_bottleneck, shape=(c_bottleneck.shape[0], target_shape[0],
#                                                    target_shape[1], 64), method='nearest')
#         u1 = nn.Conv(features=64, kernel_size=(2, 2), padding='SAME')(u1)
#         u1 = nn.relu(u1) # u1 shape: (B, 501, 501, 64)
#         u1 = jnp.concatenate([u1, c1], axis=-1)
#         c_up1 = ConvBlock(features=64)(u1, training=training)
#         output = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME')(c_up1)
#         return output

import flax.linen as nn

# Residual Conv Block
class ResBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, training: bool):
        residual = x
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        # Project residual if channels differ
        if residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1))(residual)
        x = x + residual
        x = nn.relu(x)
        return x

# Downsampling Block
class DownBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, training: bool):
        x = ResBlock(self.features)(x, training)
        skip = x
        x = nn.Conv(self.features, (3, 3), strides=(2, 2), padding="SAME")(x)
        return x, skip

# Upsampling Block
class UpBlock(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x, skip, training: bool):
        x = nn.ConvTranspose(
            self.features,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="SAME"
        )(x)
        # Handle odd sizes (e.g., 501)
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = jax.image.resize(
                x,
                (x.shape[0], skip.shape[1], skip.shape[2], x.shape[3]),
                method="bilinear",
            )
        x = jnp.concatenate([x, skip], axis=-1)
        x = ResBlock(self.features)(x, training)
        return x

# Full U-Net
class UNet(nn.Module):
    base_features: int = 32
    out_channels: int = 1
    @nn.compact
    def __call__(self, x, training: bool):
        # Encoder
        x1, skip1 = DownBlock(self.base_features)(x, training)
        x2, skip2 = DownBlock(self.base_features * 2)(x1, training)
        x3, skip3 = DownBlock(self.base_features * 4)(x2, training)
        # Bottleneck
        bottleneck = ResBlock(self.base_features * 8)(x3, training)
        # Decoder
        u3 = UpBlock(self.base_features * 4)(bottleneck, skip3, training)
        u2 = UpBlock(self.base_features * 2)(u3, skip2, training)
        u1 = UpBlock(self.base_features)(u2, skip1, training)
        # Output layer
        output = nn.Conv(self.out_channels, (1, 1), padding="SAME")(u1)
        return output

# Define the TrainState, to hold parameters and optimizer state:

class TrainState(train_state.TrainState):
    # Add batch_stats to track moving averages for Batch Normalization
    batch_stats: dict
    rng_key: jnp.ndarray

# For image-to-image regression, the Mean Squared Error (MSE) is a common and effective loss function:

def loss_fn(params, batch_stats, rng, inputs, targets, model, is_training):
    # Apply the model
    variables = {'params': params, 'batch_stats': batch_stats}
    (logits, new_model_state) = model.apply(variables, inputs, training=is_training, mutable=['batch_stats'])
    # Mean Squared Error Loss
    loss = jnp.mean(jnp.square(logits - targets))
    return loss, (logits, new_model_state)

# This function will use JAX's automatic differentiation and Optax to update the model parameters:

@jax.jit
def train_step(state, batch):
    """Performs a single training step."""
    inputs, targets = batch
    # Calculate loss and gradients
    (loss, (logits, new_model_state)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(
        state.params, state.batch_stats, state.rng_key, inputs, targets, UNet(), True
    )
    # Update Batch Normalization statistics and optimizer state
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    return new_state, loss

# Load the image pairs (In1​,Out1​) into NumPy arrays X_train and Y_train:

# find all initial condition files matching frame8****0000.png
# dir = "/Users/razoumov/training/jax/"
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

# convert lists to final NumPy arrays of shape (1000, 501, 501)
X_train = np.stack(X_list)
Y_train = np.stack(Y_list)

# add the channel dimension (1000, 501, 501) -> (1000, 501, 501, 1) for the JAX/Flax CNN model
X_train = np.expand_dims(X_train, axis=-1)
Y_train = np.expand_dims(Y_train, axis=-1)

# Set up the initial state of the model before training begins:

N, H, W, C = X_train.shape
print(f"Data shape: N={N}, H={H}, W={W}, C={C}")

MAIN_KEY = jax.random.PRNGKey(0)
key, init_key, dropout_key = jax.random.split(MAIN_KEY, 3)

# instantiate the model and dummy input
model = UNet()
dummy_input = jnp.zeros((1, H, W, C)) 

# Perform initialization, which requires a dummy input and an RNG key
init_variables = model.init(
    {'params': init_key, 'dropout': dropout_key}, 
    dummy_input, 
    training=True # Important: Use training=True for initial Batch Norm states
)

# initializes an Adam optimizer (standard optimization library for JAX; Stochastic Gradient Descent)
# learning_rate is the step size
# learning_rate=1e-4 is a very common starting point: small enough to avoid overshooting, large enough to make progress
optimizer = optax.adam(learning_rate=1e-4)

# create the initial training state
state = TrainState.create(
    apply_fn=model.apply,
    params=init_variables['params'],
    tx=optimizer,
    batch_stats=init_variables['batch_stats'],
    rng_key=dropout_key # Use the dropout key for subsequent calls
)

# Data batching:

numSamples = X_train.shape[0]
batchSize = 32
numBatchesPerEpoch = numSamples // batchSize
# Note: Samples remaining after full batches are discarded

def data_generator(X, Y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) # Shuffle indices for one pass
    for i in range(numBatchesPerEpoch):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_indices = indices[start_idx:end_idx]
        X_batch = X[batch_indices]
        Y_batch = Y[batch_indices]
        # Convert to JAX arrays before yielding
        yield jnp.asarray(X_batch), jnp.asarray(Y_batch)

numEpochs = 10
print(f"Training will run for {numEpochs} epochs, with {numBatchesPerEpoch} steps per epoch.")

# Finally, train the model:

for epoch in range(numEpochs):
    total_loss = 0.0
    # recreate the generator at the start of each epoch to ensure shuffling
    data_batches = data_generator(X_train, Y_train, batchSize)
    for step, batch in enumerate(data_batches):   # numBatchesPerEpoch iterations
        # run the JIT-compiled training step
        state, loss = train_step(state, batch)
        total_loss += loss.item() # Use .item() to get Python scalar value
        # update the PRNG key in the state (important for dropout/BN if used)
        key, new_rng = jax.random.split(state.rng_key)
        state = state.replace(rng_key=new_rng)
        # print progress every 5 steps
        if (step + 1) % 5 == 0:
            avg_loss_so_far = total_loss / (step + 1)
            print(f"  Step {step+1}/{numBatchesPerEpoch} | Running Loss: {avg_loss_so_far:.6f}")
    avg_epoch_loss = total_loss / numBatchesPerEpoch
    print(f"Epoch {epoch + 1}/{numEpochs} Complete | Average Loss: {avg_epoch_loss:.6f}")
    model_data = {'params': state.params, 'batch_stats': state.batch_stats} # data to save to disk
    # use Python's pickle for simplicity, as JAX/Flax structures are tree-like
    with open(dir+'weights%03d'%(epoch)+'.pkl', 'wb') as f: # default was `unet_model_weights.pkl`
        pickle.dump(model_data, f)

# Inference from memory:

# @jax.jit
# def predict_step(state, inputs):
#     """Performs a single inference step."""
#     variables = {'params': state.params, 'batch_stats': state.batch_stats}
#     predictions = UNet().apply(variables, inputs, training=False)
#     return predictions

# initialState = X_list[315]
# initialState = initialState.reshape(1, 501, 501, 1)
# predictedSolution = predict_step(state, initialState).reshape(501, 501)

# fig, ax = plt.subplots(figsize=(8, 8))
# cax = ax.imshow(predictedSolution, interpolation='nearest', cmap='viridis')
# plt.savefig(dir+"000.png")
# plt.close(fig)
