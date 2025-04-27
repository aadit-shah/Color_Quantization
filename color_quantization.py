import argparse
import pathlib
import time
from typing import List, Tuple, Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


def uniform_quantize(img, bits):
    if img.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    
    # Calculate bins
    step = 256 // (2**bits)  
    uniform_image = ((img // step) * step + step // 2).astype(np.uint8) 


    return uniform_image


def _init_centroids(pixels, k, rng):

    # Initialize centroids 
    n = pixels.shape[0]
    centroids = np.empty((k, 3), dtype=np.float32)

    # Pick first centroid randomly
    centroids[0] = pixels[rng.integers(0, n)]  
    for i in range(1, k):
        # Distance to nearest existing centroid
        dists = np.sum((pixels[:, None] - centroids[None, :i])**2, axis=2) 

        probs = np.min(dists, axis=1)

        # Normalize 
        probs /= probs.sum()  
        # Pick next centroid
        centroids[i] = pixels[np.searchsorted(np.cumsum(probs), rng.random())] 


    return centroids

def kmeans_quantize(img, k):

    h, w = img.shape[:2]
    # Flatten image to pixels
    pixels = img.reshape(-1, 3).astype(np.float32)  

    # Random number generator
    seed = 42
    rng = np.random.default_rng(seed)

    # Initialize centroids
    centroids = _init_centroids(pixels, k, rng)  
    max_iter = 20

    for _ in range(max_iter):

         # Compute distance to centroids
        dists = np.sum((pixels[:, None] - centroids[None, :]) ** 2, axis=2) 

        labels = np.argmin(dists, axis=1) 
        new_centroids = []

         # Update each centroid by averaging its pixels or randomly reassign if empty
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                centroid = cluster_pixels.mean(axis=0)
            else:
                random_idx = rng.integers(0, pixels.shape[0])
                centroid = pixels[random_idx]
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)

        # Break if the centroids have converged
        if np.linalg.norm(new_centroids - centroids, axis=1).max() < .0001:
            break  

        centroids = new_centroids


    final_colors = centroids.clip(0, 255).astype(np.uint8)
    return final_colors[labels].reshape(h, w, 3)  


def median_cut_quantize(img, k):
    # k must be power of 2
    if (k & (k - 1)) != 0:
        raise ValueError("k must be a power of two")  
    
    pixels = img.reshape(-1, 3)
    # Put all pixels same "box"
    boxes = [np.arange(pixels.shape[0])]  

    while len(boxes) < k:
        max_range = -1
        idx = -1
        
        # Finds the box with the biggest range in pixel color values
        for i in range(len(boxes)):
            box_pixels = pixels[boxes[i]]
            color_range = box_pixels.max(axis=0) - box_pixels.min(axis=0)
            largest_range = color_range.max()
            if largest_range > max_range:
                max_range = largest_range
                idx = i

        # Split box with largest color range
        big_box = boxes.pop(idx)  
        pts = pixels[big_box]

        # Find channel(R, G, B) with maximum range in the box
        ch = (pts.max(0) - pts.min(0)).argmax()  

        # Sort by that channel
        sorted_idx = big_box[np.argsort(pts[:, ch])]  
        mid = len(sorted_idx) // 2

        # Split into two halves
        boxes.append(sorted_idx[:mid])  
        boxes.append(sorted_idx[mid:])

    palette = []
    # Compute the mean colors for all pixels in each box to get color palette
    for box in boxes:
        box_pixels = pixels[box]

        # Average color for each box
        mean_color = box_pixels.mean(axis=0)  

        # Append average color to list
        palette.append(mean_color)

    palette = np.array(palette, dtype=np.uint8)

    # Store what box and color each pixel belongs to
    labels = np.empty(pixels.shape[0], dtype=int)
    for i, b in enumerate(boxes):
        labels[b] = i

    h, w = img.shape[:2]
    return palette[labels].reshape(h, w, 3)  # Reshape back to image


def compute_metrics(orig, quant):
    # Compute image quality metrics between original and quantized image
    return {
        "MSE": mean_squared_error(orig, quant),
        "PSNR": peak_signal_noise_ratio(orig, quant, data_range=255),
        "SSIM": structural_similarity(orig, quant, channel_axis=-1, data_range=255),
    }


def load_images(folder):
    # Load all supported images from folder
    return [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]

def save_image(path, img):
    # Save an RGB image using OpenCV (expects BGR)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def show_results(orig, results):
    # Display original and quantized images side-by-side
    plt.figure(figsize=(4 * (len(results) + 1), 4))
    plt.subplot(1, len(results) + 1, 1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis("off")
    for i, (name, img) in enumerate(results, start=2):
        plt.subplot(1, len(results) + 1, i)
        plt.imshow(img)
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # Input/Ouput Folders
    input_dir = pathlib.Path("images")  
    output_dir = pathlib.Path("results")  

    # List of quantization methods to apply
    methods = ["uniform", "kmeans", "median_cut"]  

    uniform_bits = 4  # Number of bits for uniform quantization
    kmeans_k = 16  # Number of clusters for k-means
    median_k = 256  # Number of colors for median-cut(must be power of 2)
    show = True  # Whether to display results

    # Load Images
    imgs = load_images(input_dir)
    if not imgs:
        print("No images found.")
        return
    
    # Initialize metrics collection
    metrics_summary = {m: [] for m in ["MSE", "PSNR", "SSIM"]}  

    # Loop through each image
    for img_path in imgs:
        print(f"Processing {img_path.name}:")
        bgr = cv2.imread(str(img_path)) 
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) 

        results = []
        if "uniform" in methods:
            start = time.time()
            q = uniform_quantize(rgb, uniform_bits)
            elapsed = time.time() - start
            print(f"    Uniform ({uniform_bits} bits) took {elapsed:.4f} seconds")
            results.append((f"Uniform ({uniform_bits} bits)", q))

        if "kmeans" in methods:
            start = time.time()
            q = kmeans_quantize(rgb, k=kmeans_k)
            elapsed = time.time() - start
            print(f"    K-Means (k={kmeans_k}) took {elapsed:.4f} seconds")
            results.append((f"K-Means (k={kmeans_k})", q))

        if "median_cut" in methods:
            start = time.time()
            q = median_cut_quantize(rgb, k=median_k)
            elapsed = time.time() - start
            print(f"    Median-Cut ({median_k}) took {elapsed:.4f} seconds")
            results.append((f"Median-Cut ({median_k})", q))

        # Save each image and print metrics
        for name, qimg in results:
            tag = name.split()[0].lower()  
            save_image(output_dir / f"{img_path.stem}_{tag}{img_path.suffix}", qimg)  # Save output image
            m = compute_metrics(rgb, qimg)  
            for k, v in m.items():
                metrics_summary[k].append(v)  
            print("   ", name, "|", ", ".join(f"{k}: {v:.2f}" for k, v in m.items()))
        if show:
            show_results(rgb, results)
            
    # Average metrics
    print("\n=== Average Metrics Across All Images ===")
    for k, v in metrics_summary.items():
        print(f"{k}: {np.mean(v):.2f}")  

if __name__ == "__main__":
    main()
