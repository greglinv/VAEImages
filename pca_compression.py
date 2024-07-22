# File: pca_compression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.datasets import cifar10
import time  # Add the missing import

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Apply PCA to each channel independently
n_components = 32  # Number of principal components
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
h, w, c = x_train.shape[1:]


def apply_pca(image_set, n_components):
    reconstructed_images = np.zeros_like(image_set)
    for i in range(c):
        # Flatten the image
        image_channel = image_set[:, :, :, i].reshape(len(image_set), -1)

        # Fit PCA on the channel
        pca.fit(image_channel)
        transformed = pca.transform(image_channel)

        # Inverse transform to reconstruct
        reconstructed_channel = pca.inverse_transform(transformed)
        reconstructed_images[:, :, :, i] = reconstructed_channel.reshape(image_set.shape[0], h, w)

    return reconstructed_images


# Compress and decompress the test images using PCA
pca_start_time = time.time()
x_test_reconstructed = apply_pca(x_test, n_components)
pca_end_time = time.time()
pca_time = pca_end_time - pca_start_time

# Calculate compression ratio
original_size = h * w * c
compressed_size = n_components
compression_ratio = original_size / compressed_size
print(f"PCA Compression Ratio: {compression_ratio:.2f}")

# Reshape arrays for mean squared error calculation
x_test_flat = x_test.reshape(len(x_test), -1)
x_test_reconstructed_flat = x_test_reconstructed.reshape(len(x_test_reconstructed), -1)

# Calculate reconstruction error
reconstruction_error = mean_squared_error(x_test_flat, x_test_reconstructed_flat)
print(f"PCA Reconstruction Error: {reconstruction_error:.4f}")
print(f"PCA Time: {pca_time:.2f} seconds")


# Visualize original and reconstructed images
def plot_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis("off")

        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()


plot_images(x_test, x_test_reconstructed)
