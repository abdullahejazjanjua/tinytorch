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
With warp shuffling in the forward pass:
```c
--- Config: Batch=32, Channels=256, H=56, W=56 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 0.4352 ms | PyTorch: 0.4141 ms
Backward - Custom: 0.8179 ms | PyTorch: 0.5176 ms

--- Config: Batch=128, Channels=512, H=28, W=28 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 1.0586 ms | PyTorch: 0.8250 ms
Backward - Custom: 1.0993 ms | PyTorch: 0.8852 ms

--- Config: Batch=256, Channels=1024, H=14, W=14 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 2.1748 ms | PyTorch: 0.7822 ms
Backward - Custom: 3.0771 ms | PyTorch: 0.8918 ms
```

# Softmax-CE - Fused kernel
It is important to note that this kernel only works for num_classes 64, and likely the reason the my custom kernel is faster than torch's. Also, note that I have recomputed the softmax denominator in the backward pass instead of caching in the forward pass, the reason being that I would have change the Tensor struct to allow caching, I will leave this to the future me. 
hmm, I had to write global average pooling and softmax-ce on colab because GIKI IT department can't let ensure a static IP works :(
```c
--- Config: Batch=64, Classes=10 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 0.0168 ms | PyTorch: 0.0366 ms
Backward - Custom: 0.0177 ms | PyTorch: 0.2828 ms

--- Config: Batch=1024, Classes=31 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 0.0167 ms | PyTorch: 0.0353 ms
Backward - Custom: 0.0169 ms | PyTorch: 0.2872 ms

--- Config: Batch=2048, Classes=60 ---
Forward Match:  True
Backward Match: True
Forward  - Custom: 0.0185 ms | PyTorch: 0.0372 ms
Backward - Custom: 0.0225 ms | PyTorch: 0.3026 ms

```