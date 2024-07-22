import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import time

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the VAE model
latent_dim = 64

# Encoder
encoder_inputs = tf.keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
outputs = decoder(encoder(encoder_inputs)[2])

# Define a custom layer to calculate the loss
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(inputs, reconstructed)
        )
        reconstruction_loss *= 32 * 32 * 3
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(vae_loss)
        return reconstructed

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam')

# Train the VAE
start_time = time.time()
history = vae.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))
training_time = time.time() - start_time

# Compress and Reconstruct
z_mean, z_log_var, z = encoder.predict(x_test)
x_test_decoded = decoder.predict(z)

# Calculate Metrics
compression_ratio = (np.prod(x_test.shape[1:]) * 32) / (latent_dim * 32)
reconstruction_error = np.mean(np.square(x_test - x_test_decoded))

# Output Metrics
print(f"Training Time: {training_time:.2f} seconds")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Reconstruction Error: {reconstruction_error:.6f}")

# Visualize Original and Reconstructed Images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_decoded[i])
plt.show()
