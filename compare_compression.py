# File: compare_compression.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import time
from skimage.metrics import mean_squared_error
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images to [0, 1]

# Function to compress and decompress an image using JPEG compression
def jpeg_compress_decompress(image, quality):
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer)
    return np.array(compressed_image) / 255.0  # Normalize back to [0, 1]

# Function to compress and decompress an image using DCT
def dct_compress_decompress(image, keep_fraction):
    dct_compressed = np.zeros_like(image)
    for i in range(3):
        dct = np.fft.fft2(image[:, :, i], norm='ortho')
        dct_low = np.zeros_like(dct)
        h, w = dct.shape
        dct_low[:int(h * keep_fraction), :int(w * keep_fraction)] = dct[:int(h * keep_fraction), :int(w * keep_fraction)]
        dct_compressed[:, :, i] = np.fft.ifft2(dct_low, norm='ortho').real
    return dct_compressed

# Function to compress and decompress an image using PCA
def pca_compress_decompress(image, n_components):
    h, w, c = image.shape
    pca = PCA(n_components=n_components)
    compressed = np.zeros_like(image)
    for i in range(c):
        image_channel = image[:, :, i]
        image_channel_flat = image_channel.flatten().reshape(h, w)
        pca.fit(image_channel_flat)
        transformed = pca.transform(image_channel_flat)
        inverse_transformed = pca.inverse_transform(transformed)
        compressed[:, :, i] = inverse_transformed
    return compressed

# Function to compress and decompress an image using SVD
def svd_compress_decompress(image, num_singular_values):
    channels = []
    for i in range(3):  # Iterate over the RGB channels
        U, S, V = np.linalg.svd(image[:, :, i], full_matrices=False)
        S[num_singular_values:] = 0
        channels.append(np.dot(U, np.dot(np.diag(S), V)))
    return np.stack(channels, axis=2)  # Stack the channels back together

# Measure compression ratio and reconstruction error for JPEG compression
jpeg_quality = 50
original_size = x_test[0].nbytes
jpeg_compressed_size = BytesIO()
Image.fromarray((x_test[0] * 255).astype(np.uint8)).save(jpeg_compressed_size, format="JPEG", quality=jpeg_quality)
jpeg_compressed_size = jpeg_compressed_size.tell()

jpeg_compression_ratio = original_size / jpeg_compressed_size
print(f"JPEG Compression Ratio: {jpeg_compression_ratio:.2f}")

# Compress and decompress the test images and measure time
jpeg_start_time = time.time()
jpeg_reconstructed_imgs = np.array([jpeg_compress_decompress(img, jpeg_quality) for img in x_test])
jpeg_end_time = time.time()
jpeg_time = jpeg_end_time - jpeg_start_time

# Measure normalized reconstruction error for JPEG compression
jpeg_reconstruction_error = mean_squared_error(x_test, jpeg_reconstructed_imgs)
print(f"JPEG Reconstruction Error: {jpeg_reconstruction_error:.4f}")
print(f"JPEG Time: {jpeg_time:.2f} seconds")

# DCT Compression
keep_fraction = 0.1
dct_start_time = time.time()
dct_compressed_imgs = np.array([dct_compress_decompress(img, keep_fraction) for img in x_test])
dct_end_time = time.time()
dct_time = dct_end_time - dct_start_time
dct_compression_ratio = 1 / keep_fraction ** 2
dct_reconstruction_error = mean_squared_error(x_test, dct_compressed_imgs)
print(f"DCT Compression Ratio: {dct_compression_ratio:.2f}")
print(f"DCT Reconstruction Error: {dct_reconstruction_error:.4f}")
print(f"DCT Time: {dct_time:.2f} seconds")

# PCA Compression
n_components = 32
pca_start_time = time.time()
pca_compressed_imgs = np.array([pca_compress_decompress(img, n_components) for img in x_test])
pca_end_time = time.time()
pca_time = pca_end_time - pca_start_time
pca_compression_ratio = original_size / (n_components * x_test[0].shape[0] * x_test[0].shape[1])
pca_reconstruction_error = mean_squared_error(x_test, pca_compressed_imgs)
print(f"PCA Compression Ratio: {pca_compression_ratio:.2f}")
print(f"PCA Reconstruction Error: {pca_reconstruction_error:.4f}")
print(f"PCA Time: {pca_time:.2f} seconds")

# SVD Compression
num_singular_values = 100
svd_start_time = time.time()
svd_compressed_imgs = np.array([svd_compress_decompress(img, num_singular_values) for img in x_test])
svd_end_time = time.time()
svd_time = svd_end_time - svd_start_time
svd_compression_ratio = original_size / (num_singular_values * (x_test[0].shape[0] + x_test[0].shape[1]))
svd_reconstruction_error = mean_squared_error(x_test, svd_compressed_imgs)
print(f"SVD Compression Ratio: {svd_compression_ratio:.2f}")
print(f"SVD Reconstruction Error: {svd_reconstruction_error:.4f}")
print(f"SVD Time: {svd_time:.2f} seconds")

# Visualize original and compressed images
def plot_images(original_images, compressed_images, n, title):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.axis("off")
    for i in range(n):
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(compressed_images[i])
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

plot_images(x_test[:10], jpeg_reconstructed_imgs[:10], 10, "JPEG Compression")
plot_images(x_test[:10], dct_compressed_imgs[:10], 10, "DCT Compression")
plot_images(x_test[:10], pca_compressed_imgs[:10], 10, "PCA Compression")
plot_images(x_test[:10], svd_compressed_imgs[:10], 10, "SVD Compression")

# Placeholder for your VAE/DEC comparison results
# Print your VAE/DEC compression ratio and reconstruction error here
# Example:
# vae_compression_ratio = ...
# vae_reconstruction_error = ...
# print(f"VAE Compression Ratio: {vae_compression_ratio:.2f}")
# print(f"VAE Reconstruction Error: {vae_reconstruction_error:.4f}")
