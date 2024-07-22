# File: main.py

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import EarlyStopping

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define VAE Model
latent_dim = 128  # Increased latent dimension

# Encoder
encoder_inputs = tf.keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)  # Added layer
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)  # Increased dense layer size
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(4 * 4 * 256, activation="relu")(latent_inputs)  # Adjusted for increased latent size
x = layers.Reshape((4, 4, 256))(x)  # Adjusted for increased latent size
x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)  # Added layer
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)  # Adjusted strides to match dimensions
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)

decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE Model
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, vae_outputs, name="vae")

# VAE Loss within a custom layer
class VAEModel(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        self.add_loss(reconstruction_loss + kl_loss)
        return reconstructed

vae = VAEModel(encoder, decoder)
vae.compile(optimizer=optimizers.Adam(learning_rate=0.001))  # Adjusted learning rate

# Train VAE
vae.fit(x_train, x_train, epochs=2, batch_size=64, validation_data=(x_test, x_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)])  # Increased epochs and patience

# Measure Compression Ratio
original_size = np.prod(x_train.shape[1:])
compressed_size = latent_dim
compression_ratio = original_size / compressed_size
print(f"Compression Ratio: {compression_ratio:.2f}")

# Measure Reconstruction Error
reconstructed_imgs = vae.predict(x_test)
reconstruction_error = np.mean((x_test - reconstructed_imgs) ** 2)
print(f"Reconstruction Error: {reconstruction_error:.4f}")

# Initialize cluster centers using k-means
encoder_model = models.Model(encoder_inputs, z_mean)
latent_space = encoder_model.predict(x_train)
kmeans = KMeans(n_clusters=10, n_init=20)
y_pred = kmeans.fit_predict(latent_space)
cluster_centers = kmeans.cluster_centers_

# DEC Model
class DECModel(tf.keras.Model):
    def __init__(self, encoder, cluster_centers, **kwargs):
        super(DECModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.cluster_centers = tf.Variable(cluster_centers, dtype=tf.float32, trainable=False)

    def target_distribution(self, q):
        weight = q ** 2 / tf.reduce_sum(q, axis=0)
        return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, axis=1))

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        q = 1.0 / (1.0 + tf.reduce_sum(tf.square(tf.expand_dims(z, axis=1) - self.cluster_centers), axis=2))
        q = q ** ((1.0 + 1.0) / 2.0)
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        p = self.target_distribution(q)
        return q, p

# Initialize cluster centers
dec = DECModel(encoder, cluster_centers)
dec.compile(optimizer=optimizers.Adam(learning_rate=0.01))

# Custom training loop for DEC
@tf.function
def train_step(model, inputs):
    with tf.GradientTape() as tape:
        q, p = model(inputs)
        kl_loss = tf.reduce_sum(p * tf.math.log(p / q), axis=-1)
        loss = tf.reduce_mean(kl_loss)
    grads = tape.gradient(loss, model.encoder.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.encoder.trainable_weights))
    return loss

# Train DEC Model
epochs = 2
batch_size = 128
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i:i + batch_size]
        loss = train_step(dec, batch)
        epoch_loss += loss.numpy()
    print(f"Loss: {epoch_loss / (len(x_train) / batch_size)}")

# Visualize results
def plot_images(images, n):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

plot_images(x_test[:10], 10)
plot_images(reconstructed_imgs[:10], 10)

# Visualize latent space with labeled axes
tsne = TSNE(n_components=2)
latent_2d = tsne.fit_transform(latent_space)
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=y_pred)
plt.xlabel('t-SNE component 1')  # Label for x-axis
plt.ylabel('t-SNE component 2')  # Label for y-axis
plt.title('t-SNE Visualization of Latent Space')  # Optional title
plt.show()
