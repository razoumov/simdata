class UNetBlock(nnx.Module):
    def __init__(self, in_chan, out_chan, rngs: nnx.Rngs, stride=1):
        self.conv = nnx.Conv(in_chan, out_chan, kernel_size=(3, 3), strides=stride, padding='SAME', rngs=rngs)
        self.bn = nnx.BatchNorm(out_chan, momentum=0.9, rngs=rngs)
    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nnx.relu(x)

class UNet(nnx.Module):
    def __init__(self, in_features, out_features, rngs: nnx.Rngs):
        # Encoder: 32 -> 64 -> 128 -> 256
        self.enc1 = UNetBlock(in_features, 32, rngs)
        self.enc2 = UNetBlock(32, 64, rngs, stride=2)
        self.enc3 = UNetBlock(64, 128, rngs, stride=2)
        self.enc4 = UNetBlock(128, 256, rngs, stride=2)
        # Bottleneck: capped at 512
        self.bottleneck = UNetBlock(256, 512, rngs)
        # Decoder
        self.up4 = nnx.ConvTranspose(512, 256, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec4 = UNetBlock(512, 256, rngs) # (256 up + 256 skip)
        self.up3 = nnx.ConvTranspose(256, 128, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec3 = UNetBlock(256, 128, rngs) # (128 up + 128 skip)
        self.up2 = nnx.ConvTranspose(128, 64, kernel_size=(3, 3), strides=2, padding='SAME', rngs=rngs)
        self.dec2 = UNetBlock(128, 64, rngs)  # (64 up + 64 skip)
        self.final_conv = nnx.Conv(64, out_features, kernel_size=(1, 1), rngs=rngs)
    def __call__(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        # Helper to handle the shape matching for skip connections
        def upsample_and_concat(current, skip):
            # current.shape[0] is batch, skip.shape[1:3] is HW, current.shape[-1] is C
            target_shape = (current.shape[0], skip.shape[1], skip.shape[2], current.shape[3])
            up = jax.image.resize(current, target_shape, method="bilinear")
            return jnp.concatenate([up, skip], axis=-1)
        # Decoder path
        x = upsample_and_concat(self.up4(b), s4)
        x = self.dec4(x)
        x = upsample_and_concat(self.up3(x), s3)
        x = self.dec3(x)
        x = upsample_and_concat(self.up2(x), s2)
        x = self.dec2(x)
        # Final resize to exactly 500x500 before the last conv
        final_shape = (x.shape[0], 500, 500, x.shape[3])
        x = jax.image.resize(x, final_shape, method="bilinear")        
        return jax.nn.sigmoid(self.final_conv(x))
