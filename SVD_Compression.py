import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0  # Normalize images
x_test = x_test / 255.0

# Define SVD compression function
def svd_compress(image, k):
    u, s, vh = np.linalg.svd(image, full_matrices=False)
    u_k = u[:, :k]
    s_k = np.diag(s[:k])
    vh_k = vh[:k, :]
    compressed_image = np.dot(u_k, np.dot(s_k, vh_k))
    return compressed_image

# Define PCA compression function
def pca_compress(image, k):
    compressed_image = np.zeros_like(image)
    for i in range(3):  # Apply PCA on each color channel
        img_channel = image[:, :, i]
        pca = PCA(n_components=k)
        img_channel_reduced = pca.fit_transform(img_channel)
        img_channel_reconstructed = pca.inverse_transform(img_channel_reduced)
        compressed_image[:, :, i] = img_channel_reconstructed
    return compressed_image

# Apply compression methods to dataset and calculate metrics
def apply_compression_and_evaluate(images, k):
    results = {
        'svd': {'compression_ratios': [], 'reconstruction_errors': [], 'total_time': 0},
        'pca': {'compression_ratios': [], 'reconstruction_errors': [], 'total_time': 0}
    }

    # SVD Compression
    start_time = time.time()
    for img in images:
        compressed_img = np.zeros_like(img)
        for i in range(3):
            compressed_img[:, :, i] = svd_compress(img[:, :, i], k)

        compression_ratio = (k * (img.shape[0] + img.shape[1] + 1)) / (img.shape[0] * img.shape[1])
        results['svd']['compression_ratios'].append(compression_ratio)

        img_flat = img.flatten()
        compressed_img_flat = compressed_img.flatten()
        reconstruction_error = mean_squared_error(img_flat, compressed_img_flat)
        results['svd']['reconstruction_errors'].append(reconstruction_error)
    results['svd']['total_time'] = time.time() - start_time

    # PCA Compression
    start_time = time.time()
    for img in images:
        compressed_img = pca_compress(img, k)

        compression_ratio = (k * (img.shape[0] + img.shape[1] + 1)) / (img.shape[0] * img.shape[1])
        results['pca']['compression_ratios'].append(compression_ratio)

        img_flat = img.flatten()
        compressed_img_flat = compressed_img.flatten()
        reconstruction_error = mean_squared_error(img_flat, compressed_img_flat)
        results['pca']['reconstruction_errors'].append(reconstruction_error)
    results['pca']['total_time'] = time.time() - start_time

    avg_results = {
        'svd': {
            'avg_compression_ratio': np.mean(results['svd']['compression_ratios']),
            'avg_reconstruction_error': np.mean(results['svd']['reconstruction_errors']),
            'total_time': results['svd']['total_time']
        },
        'pca': {
            'avg_compression_ratio': np.mean(results['pca']['compression_ratios']),
            'avg_reconstruction_error': np.mean(results['pca']['reconstruction_errors']),
            'total_time': results['pca']['total_time']
        }
    }

    return avg_results

# Set rank k for SVD and PCA
k = 20

# Evaluate on a subset of training images for demonstration purposes
subset_size = 1000
avg_results = apply_compression_and_evaluate(x_train[:subset_size], k)

print(f'SVD Compression:')
print(f'  Average Compression Ratio: {avg_results["svd"]["avg_compression_ratio"]:.4f}')
print(f'  Average Reconstruction Error: {avg_results["svd"]["avg_reconstruction_error"]:.4f}')
print(f'  Total Time Taken: {avg_results["svd"]["total_time"]:.2f} seconds')

print(f'PCA Compression:')
print(f'  Average Compression Ratio: {avg_results["pca"]["avg_compression_ratio"]:.4f}')
print(f'  Average Reconstruction Error: {avg_results["pca"]["avg_reconstruction_error"]:.4f}')
print(f'  Total Time Taken: {avg_results["pca"]["total_time"]:.2f} seconds')
