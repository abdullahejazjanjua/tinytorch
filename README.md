# TinyTorch

This repository contains a tiny implementation of the torch library.

# Convolution Kernel
Implemented Forward pass, backward pass wrt input, weight.

## Convolution Kernel Limitations

The current implementation of the 2D convolution engine is optimized for basic model training (e.g., LeNet, VGG) but operates under several technical and hardware constraints.

### 1. Geometric & Architectural Constraints

* **Kernel Sizes**: The backward weight pass supports only square kernels of sizes **1, 3, 5, and 7**. Even-sized kernels or dimensions larger than 7 will trigger a runtime error.
* **Stride and Dilation**: Only **Stride 1** and **Dilation 1** are natively supported. The current shared memory loading and indexing logic do not account for skipping input pixels or expanding the kernel receptive field.
* **Convolution Variants**: Supports standard 2D convolution only. There is no support for grouped, depthwise separable, or transposed convolutions.
* **Padding**: Supports either "Same" padding (calculated as $\frac{K-1}{2}$) or "Valid" padding (0).

### 2. Grid & Dimensional Limits

* **Hardware Grid Limit**: In the backward input pass, the grid Z-dimension is calculated as $N \times C_{in}$. Due to NVIDIA hardware limits on `gridDim.z`, the product of $BatchSize \times InputChannels$ must not exceed **65,535**.
* **Resource Occupancy**: The backward weight kernel uses a hardcoded block size of **16** to manage register pressure and local array allocation for $K \times K$ filters.

### 3. Numerical & Initialization Requirements

* **Precision Drift**: The kernels utilize naive **FP32** accumulation. In configurations with high channel depths ($C_{in} \ge 128$) or large kernels ($K=7$), floating-point rounding errors may result in a relative difference of up to $10^{-3}$ compared to cuDNN due to different summation orders.

## Global Pooling
```C
--- Config: Batch=32, Channels=256, H=56, W=56 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 0.5461 ms | PyTorch: 0.4157 ms
Backward - Custom: 0.8275 ms | PyTorch: 0.4448 ms

--- Config: Batch=128, Channels=512, H=28, W=28 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 1.0481 ms | PyTorch: 0.8189 ms
Backward - Custom: 1.1399 ms | PyTorch: 0.8860 ms

--- Config: Batch=256, Channels=1024, H=14, W=14 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 3.1151 ms | PyTorch: 0.7788 ms
Backward - Custom: 2.9275 ms | PyTorch: 0.8946 ms
```