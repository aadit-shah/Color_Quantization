# 📄 README: Image Quantization Project

## Overview

This project implements and compares **three different color quantization algorithms**:

- **Uniform Quantization**
- **K-Means Quantization**
- **Median-Cut Quantization**

This project focuses on **implementing and evaluating classical color quantization algorithms without external machine learning libraries**.

The goal of this assignment is to **compress color images** by reducing the number of unique colors while maintaining visual quality, and **analyze** the trade-offs between simplicity, performance, and quality.

We also evaluate the results using standard image similarity metrics:  
**MSE (Mean Squared Error)**, **PSNR (Peak Signal-to-Noise Ratio)**, and **SSIM (Structural Similarity Index)**.

---

## ✨ How Each Algorithm Works

### 1. **Uniform Quantization**
- The pixel values are uniformly divided into bins.
- Each pixel's color is mapped to the nearest bin center.
- This is the **fastest** and simplest method, but it does **not adapt** to the content of the image.
- **Parameter**:  
  - `uniform_bits` = number of bits to use per channel (e.g., 4 bits → 16 levels per channel)

---

### 2. **K-Means Quantization**
- Clusters the pixel colors into `k` groups based on color similarity.
- Each pixel is replaced with the centroid of its assigned cluster.
- **K-Means++** initialization is used to improve cluster seeding.
- This method gives **adaptive** results based on the image’s color distribution.
- **Parameter**:
  - `kmeans_k` = number of color clusters to use (e.g., 16)

---

### 3. **Median-Cut Quantization**
- Recursively splits the set of pixels along the color channel with the largest range.
- Each box (group of colors) is split until `k` boxes are formed.
- The average color of each box becomes the representative color.
- Median-cut is **efficient** and **content-aware** but less precise than k-means clustering.
- **Parameter**:
  - `median_k` = number of final colors (must be a power of 2, e.g., 256)

---

## 🎯 Why These Methods Were Chosen

- **Uniform Quantization**: Provides a **simple baseline**. It's fast and easy to implement but does not adapt to the image structure.
- **K-Means Quantization**: Represents a **high-quality adaptive method** that often gives the best looking images at the cost of higher computation time.
- **Median-Cut Quantization**: A **classic algorithm** that balances speed and adaptiveness, widely used historically (e.g., in GIF compression).

Together, these three algorithms give a **good spectrum** from simple to more complex, and from fast to more accurate.

---

## ⚙️ How to Run

### 1. Setup

Make sure you have the following Python packages installed:

```bash
pip install numpy opencv-python matplotlib scikit-image
```

### 2. Organize Files

- Put your input images in a folder named `images/` (in the same directory as the script).
- Example:

```
project-folder/
  images/
    img1.jpg
    img2.png
  your_script.py
```

### 3. Run the script

In the terminal:

```bash
python your_script.py
```

The quantized images will be saved into a folder called `results/`, and you will see a side-by-side display of results for each image.

Metrics for each method and the average across all images will be printed in the terminal.

---

## 🛠️ How to Modify

You can easily adjust parameters at the top of the `main()` function:

| Parameter   | What it controls                              | Example Change                |
|:-----------:|:----------------------------------------------|:-------------------------------|
| `uniform_bits` | Number of bits in uniform quantization       | Set `uniform_bits = 3` for coarser quantization |
| `kmeans_k`     | Number of clusters in k-means quantization   | Set `kmeans_k = 32` for finer clustering |
| `median_k`     | Number of colors for median-cut (power of 2) | Set `median_k = 128` for fewer colors |

You can also modify `methods` to choose which quantization algorithms to apply:

```python
methods = ["uniform", "kmeans"]  # Only apply Uniform and K-Means
```

Or turn off the plot display:

```python
show = False  # Disable visualization
```

---

## 📊 Metrics Explained

- **MSE (Mean Squared Error)**: Lower is better.
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better.
- **SSIM (Structural Similarity Index)**: Closer to 1 is better.

These give a **quantitative comparison** of how much visual distortion is introduced by each quantization method.

---

# ✅ Conclusion

This project shows how different color quantization methods perform in terms of speed and image quality.  
Each method has its strengths and weaknesses depending on the application:

- Use **Uniform Quantization** for speed and simplicity.
- Use **K-Means** for best quality but more compute.
- Use **Median-Cut** for a good balance when fast adaptive compression is needed.

---
