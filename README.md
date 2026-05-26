# TinyTorch

TinyTorch is a minimalist deep learning library we built to better understand how these frameworks actually function under the hood. The internal logic and Python API are heavily inspired by PyTorch, but the codebase is kept minimal and CUDA-native. There is no CPU fallback here because the goal was a lean implementation that focuses entirely on NVIDIA GPUs. We included 2D linear and convolution layers, ReLU activations, a fused softmax-cross-entropy loss for better efficiency, and an SGD optimizer because it was the simplest to implement. The library also features a complete custom autograd system for backpropagation. The code is designed to be easily navigational, so as long as you have a decent understanding of C++ and CUDA, you can read through it and see exactly how everything connects. As a proof of concept, the model currently hits 95% accuracy on the MNIST test set.

---

> *"What I cannot create, I do not understand."*
>
> - Richard Feynman

## Repository layout
*   `cuda/`: Forward and backward kernels for each operation alongside a host-side kernel launcher.
*   `src/nn`: Class definitions for each layer that initialize weights and parameters (like kernel size) and implement a forward method to call the corresponding `src/functional` wrapper.
*   `src/functional`: Wrappers that invoke the host-side kernel launcher and initialize nodes with appropriate parameters.
*   `src/optim`: The SGD optimizer class definition, kept separate to allow for easy extension with additional optimizers.
*   `src/autograd`: Logic for topological sorting of the constructed graph and calling backward functions in reverse order.
*   `include`: Header files for class and function declarations.
*   `mnist-dataloader`: The dataloader and helper functions for MNIST; because the dataset is small, it is kept entirely in RAM.
*   `bind/`: Defines three separate modules and binds them to Python using PyBind11.


## Prerequisites

First, you will need an NVIDIA GPU. If you do not have one, you can use the one in Colab; just run the following commands. It should work, although I haven't tested it.

You must have the following installed on your system:
- CMake >= 3.18
- pybind11 (I don't know which version; just run `pip install pybind11`)
- nvcc compiler
- clang++ or g++

## Building from source

Now that you have the requirements installed, run the following commands:

```text
cd bind
mkdir build
cd build
cmake ..
cmake --build .
```

It is important to note that I have only tested this on Linux (I mean, who uses an NVIDIA GPU on Windows? Surely not you).

This will create four modules in the root of this project: `mnist_io`, `optim`, `base`, and `nn`.

## Tensors

The tensor is declared in `include/tensor.h` as:
```cpp
typedef struct Tensor {
    float *data;        // 1D pointer to the data
    int ndim;           // Number of dimensions
    int *shape;         // Array containing the shape of the tensor
    int size;           // Total size of data (not in bytes)
    Node *prev;         // Node linking this tensor to its parent tensors in the computational graph

    int on_gpu;         // Flag indicating if data is on the GPU or CPU
    int requires_grad;  // Flag to enable/disable gradient computation
    struct Tensor *grad; // Pointer to the gradient tensor
} Tensor;
```

## GPU Resident Tensors
The output of each layer is kept in GPU VRAM by default to avoid the overhead of moving data between the CPU and GPU.  An important optimization we made was keeping the tensor metadata on the CPU; because we pass these members to CUDA kernels by value, we don't need to store the entire `Tensor` struct on the GPU.

## Modules

### mnist_io

* `mnist_io.load_dataset_in_ram(images_path: str, labels_path: str)`: Loads the entire MNIST dataset into system memory for fast access.
* `mnist_io.load_batch_to_tensor(data: MNISTData, batch_size: int, indices: list[int], offset: int, images_tensor: Tensor, labels_tensor: Tensor)`: Populates image and label tensors from a specific batch of the dataset.
* `mnist_io.create_indices(num_samples: int)`: Generates a list of integers used to track and shuffle data access.


### base

* `base.tensor_create(shape: list[int], requires_grad: bool)`: Factory function to instantiate tensors with specified dimensions and gradient tracking.
* `base.tensor_to_gpu(tensor: Tensor)`: Migrates tensor data from host (CPU) memory to device (GPU) memory.
* `base.tensor_to_cpu(tensor: Tensor)`: Migrates tensor data from device (GPU) memory back to host (CPU) memory.


### nn

* `nn.Conv2D(in_channels: int, out_channels: int, kernel_size: int, padding: int, requires_grad: bool)`: Class constructor for a 2D convolution layer specifying input/output channels and spatial parameters.
* `nn.Linear(in_features: int, out_features: int, requires_grad: bool, use_bias: bool)`: Class constructor for a fully connected layer with input/output feature sizes and optional bias.
* `nn.ReLU()`: Class constructor for the Rectified Linear Unit activation module.
* `nn.GlobalPooling()`: Class constructor for the spatial reduction module (Global Average Pooling).
* `nn.CrossEntropy()`: Class constructor for the loss module used in multi-class classification tasks.


### optim

* `optim.SGD(params_list: list[Tensor], lr: float)`: Class constructor for the Stochastic Gradient Descent optimizer, managing a list of parameters.
* `optim.SGD.zero_grad()`: Method to reset the accumulated gradients of all tensors managed by the optimizer instance to zero.
* `optim.backward(loss_tensor: Tensor)`: Global function that initiates the autograd engine's topological sort and backpropagation starting from the provided loss tensor.

## Limitations

We have an "adoption" policy because we do not leave any orphaned tensors behind. When a layer returns a newly allocated tensor, you effectively "adopt" it and assume full responsibility for its life cycle. Concretely, you must manually free all tensors using the utility function `base.tensor_free(Tensor *)`. A simple destructor is insufficient here because it cannot account for the complex dependencies within the computational graph. A proper automated solution would require a dedicated garbage collector, which is out of scope for this project. Now you have seen manual memory management in Python (go ahead, laugh).

### Why create on CPU (for dataloading) then `tensor_to_gpu`

Although `tensor_create` allows for direct GPU allocation, the resulting memory view is awkward to populate from NumPy across the PCIe bus. We instead allocate tensors on the CPU first to facilitate moving images sequentially from RAM into the tensor before transferring the entire batch to the GPU in one go using `base.tensor_to_gpu(x)`. Layers like `Conv2D` and `Linear` still allocate their weights directly on the GPU within their constructors to minimize host-to-device overhead.


## Python API


```python
import numpy as np
import base
import nn
import optim as optim_mod

N, in_features, num_classes = 16, 64, 10 
lr = 0.01

# Tensors are created on CPU first to allow sequential loading from RAM
x = base.tensor_create([N, in_features], 1, 0)
x.data[:] = np.random.standard_normal(N * in_features).astype(np.float32)
base.tensor_to_gpu(x)

labels = base.tensor_create([N], 0, 0)
labels.data[:] = np.random.randint(0, num_classes, size=N).astype(np.float32)
base.tensor_to_gpu(labels)

# Initialize layers and optimizer
# Weights for Linear layers are allocated on the GPU within the constructor
lin = nn.Linear(in_features, num_classes, 1, 1)
optimizer = optim_mod.SGD([lin.weights, lin.bias], lr)
ce = nn.CrossEntropy(1)

# Forward pass
# Output tensors stay on the GPU to avoid movement overhead
logits = lin.forward(x)
loss = ce.forward(logits, labels)

# Backward and update
optimizer.zero_grad()  # Reset gradients
optim_mod.backward(loss) # compute gradients for all layer weights
optimizer.step()       # Must update weights before clearing gradients

# Transfer back to CPU to inspect results
base.tensor_to_cpu(loss)
print(f"Loss: {loss.data[0]:.4f}")

# Manually free all "adopted" tensors to prevent memory leaks
base.tensor_free(loss)
base.tensor_free(logits)
base.tensor_free(x)
base.tensor_free(labels)
```

You can find a sample training code on mnist in `train-mnist.py`

---

## Hardware and API limitations (convolution in depth)

The convolution implementation in `cuda/conv.cu` is written for training small CNNs, not for general computer vision at arbitrary batch width and channel width. Limits come from template choices, shared memory tiling, and CUDA’s maximum grid dimensions.

### Kernel sizes, stride, padding, and what kind of conv

*   Backward w.r.t. weights is instantiated only for square odd kernels 1, 3, 5, and 7. Other sizes either will not compile the right template or will hit a default branch that does not launch a kernel.
*   There is one spatial conv: no groups, no depthwise separable, and no transposed convolution.
*   Padding is either "same" in the sense that the output height and width match the input. For valid convolution you need the input spatial size large enough that H_out = H - K + 1 (and similarly for W) is positive.

### Occupancy and numerics

The backward-weight kernel uses a fixed block size of 16 and keeps a small on-chip array sized for K x K filters. That is a deliberate trade to control register pressure; it is not tuning for every GPU generation.

All training here is in FP32 with "do the obvious multiply-add loop" semantics. Compared to PyTorch, which may reorder reductions and/or use different accumulation strategies, you can see relative errors on the order of 10^-3 on unfriendly shapes. That is expected for different floating-point trees, not necessarily a bug.

### Global average pooling

Earlier versions of the global pooling kernel ran into grid limits when batch x channels grew large, similar in spirit to the conv backward-input issue. The current pooling path was reworked so that large N x C products are handled without leaning on an oversized gridDim.z. Pooling is still meant for reasonably sized feature maps in small CNNs; it is not a general ND reduction framework.

### Fused softmax + cross-entropy

The fused loss kernel is only valid when num_classes is strictly less than 64, meaning you would practically use 1 through 63 output logits. The implementation assumes a class dimension small enough that a single warp can cooperate during reduction. This choice was made because most of our target datasets, such as MNIST, have fewer than 64 classes. While we originally recomputed quantities in the backward path to avoid adding extra fields to the Tensor struct, the logic was later updated to allow the Node to store the necessary context for backward calls. This change makes the previous justification for redundant math moot and identifies a potential area for optimization. 

## Performance Comparison: MNIST Training

The following numbers come from one pair of runs on the same machine, same data root, same architecture description, and same optimization hyperparameters, differing only in whether the loop was driven by TinyTorch+IDX or by PyTorch+torchvision MNIST tensors.

### Setup Parameters

| Parameter | Value |
| :--- | :--- |
| **Training Samples** | 59,968 (Remainder of 60,000 dropped for batch size consistency) |
| **Batch Size** | 64 |
| **Learning Rate** | 0.05 |
| **Epochs** | 10 |
| **Test Samples** | 9,984 (Full batches only) |


### Architecture Configuration

The network architecture is identical for both frameworks:

*   **Convolutional Stem:** 1 → 48 → 96 channels using $5 \times 5$ convolutions.
*   **Padding:** Matches the TinyTorch convention.
*   **Global Average Pooling:** Collapses spatial dimensions to a vector of width 96.
*   **MLP Head:** 96 → 128 → 10 layers with ReLU activations.


### Data and Evaluation

*   **Data Handling:** TinyTorch reads raw IDX files, while PyTorch uses the `torchvision` MNIST dataset rooted at the same directory.
*   **Evaluation Protocol:** Testing uses 9,984 samples to maintain full batch counts.

## Summary Table


| Metric | TinyTorch | PyTorch (CUDA) |
| :--- | :--- | :--- |
| Total training wall time | 1261 s (~21 min) | 64 s |
| Mean train loss (epoch 10) | 0.183 | 0.308 |
| Train accuracy (epoch 10) | 94.5% | 91.0% |
| Test: mean loss | 0.143 | 0.205 |
| Test: accuracy | 95.6% | 94.2% |


TinyTorch shows lower final training loss and higher training accuracy on this split, while its test accuracy is also slightly higher. These results are from single seeds and runs without exhaustive sweeps; performance differences may stem from initialization details, numerical ordering, or data order, among other factors. Conversely, PyTorch trains the model approximately 20x faster because it utilizes highly optimized kernels.

## Layer micro-benchmarks (full table and interpretation)

This section records a single micro-benchmark session so you can see raw medians, not cherry-picked anecdotes.

### Environment

| Field | Value |
| :--- | :--- |
| GPU | NVIDIA RTX A2000 12GB |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| Report timestamp | 2026-05-02T04:34:17 |

### Timing protocol

For each row below, the harness measures one forward pass and one backward pass through the relevant operator. This includes a synthetic upstream gradient where the output is not a scalar, exactly like how training would eventually propagate a gradient into that tensor. Times are median milliseconds over 80 timed iterations after 20 warmup iterations. Each timed iteration is wrapped with CUDA device synchronization so asynchronous launches do not misrepresent which pass finished.

PyTorch’s side had TensorFloat-32 disabled for both cuBLAS matmul and cuDNN so the comparison stays in an FP32 spirit similar to the TinyTorch kernels.

The column Torch/Tiny is the ratio (PyTorch median ms) / (TinyTorch median ms). If it is less than 1, PyTorch was faster on that row (it took fewer milliseconds). If it is greater than 1, TinyTorch was faster.

Between layer families, such as after finishing all convolution rows and before starting linear rows, the harness also synchronized, cleared the CUDA allocator cache, and summed a ~96 MiB scratch tensor once. This is not a feature of normal training; it is used so the cold-cache behavior of the next family is less contaminated by the L2 state of the previous family. Individual rows within a family were not separated by that flush; only broad section boundaries were.

All 49 configurations launched successfully. Each configuration respected the convolution grid constraints documented earlier in this README.

### Conv2d (18 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=2 Ci=8 HW=32×32 Co=16 k=3 same | 0.7215 | 0.3843 | 0.533 |
| N=4 Ci=16 HW=32×32 Co=32 k=3 same | 2.2762 | 0.4869 | 0.214 |
| N=8 Ci=32 HW=28×28 Co=64 k=5 same | 14.6211 | 1.0178 | 0.070 |
| N=16 Ci=64 HW=28×28 Co=64 k=3 same | 17.4406 | 1.1630 | 0.067 |
| N=16 Ci=1 HW=28×28 Co=8 k=5 same | 0.5625 | 0.5444 | 0.968 |
| N=32 Ci=128 HW=16×16 Co=256 k=3 same | 112.2537 | 2.4035 | 0.021 |
| N=32 Ci=128 HW=16×16 Co=32 k=3 valid | 14.8683 | 0.5518 | 0.037 |
| N=64 Ci=256 HW=14×14 Co=255 k=3 same | 413.0922 | 6.9573 | 0.017 |
| N=4 Ci=64 HW=14×14 Co=128 k=7 valid | 22.2423 | 0.4831 | 0.022 |
| N=32 Ci=512 HW=7×7 Co=127 k=3 same | 179.4979 | 1.6619 | 0.009 |
| N=1 Ci=3 HW=224×224 Co=64 k=7 same | 92.1635 | 2.8597 | 0.031 |
| N=2 Ci=3 HW=224×224 Co=64 k=7 same | 103.2279 | 5.2425 | 0.051 |
| N=4 Ci=64 HW=56×56 Co=128 k=3 same | 71.9913 | 2.0725 | 0.029 |
| N=8 Ci=128 HW=56×56 Co=256 k=3 same | 407.8909 | 9.0796 | 0.022 |
| N=16 Ci=256 HW=28×28 Co=255 k=3 same | 331.3403 | 7.8077 | 0.024 |
| N=1 Ci=256 HW=8×8 Co=48 k=1 same | 1.6799 | 0.3100 | 0.185 |
| N=8 Ci=48 HW=32×32 Co=96 k=5 same | 48.0488 | 2.8455 | 0.059 |
| N=256 Ci=64 HW=8×8 Co=128 k=3 same | 215.2738 | 3.7265 | 0.017 |

#### Discussion (convolution)

On almost every line, PyTorch finishes the fused forward-backward pair in a fraction of the time TinyTorch needs. The ratio is least punishing on tiny problems, such as single-channel MNIST-ish maps with few output channels. In these instances, fixed launch and synchronization overhead constitutes a measurable slice of TinyTorch's time. This is evident in the row with Ci=1 and a Torch/Tiny ratio of ~0.97.

As soon as width, height, and channels grow, TinyTorch times climb into tens or hundreds of milliseconds per step while PyTorch stays in single digits for comparable geometry. The 224x224 stem-style rows with k = 7 show that spatial size impacts the teaching kernel much more severely than a vendor library that has spent years optimizing occupancy and memory traffic. This is the expected result of not implementing a full autotuning convolution factory.

### Linear (12 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| batch=1 in=4096 out=4096 | 47.9943 | 0.9071 | 0.019 |
| batch=4 in=4096 out=4096 | 48.0997 | 0.9158 | 0.019 |
| batch=8 in=3584 out=3584 | 37.0349 | 0.8754 | 0.024 |
| batch=16 in=1024 out=4096 | 6.8094 | 0.5830 | 0.086 |
| batch=32 in=768 out=3072 | 4.4649 | 0.4618 | 0.103 |
| batch=64 in=512 out=2048 | 2.7679 | 0.4896 | 0.177 |
| batch=128 in=784 out=256 | 1.0985 | 0.4386 | 0.399 |
| batch=256 in=512 out=512 | 2.0405 | 0.5960 | 0.292 |
| batch=512 in=256 out=1024 | 3.2088 | 0.7870 | 0.245 |
| batch=1024 in=128 out=512 | 2.3408 | 0.7261 | 0.310 |
| batch=64 in=2048 out=512 | 2.7471 | 0.5429 | 0.198 |
| batch=32 in=8192 out=2048 | 50.2132 | 1.5569 | 0.031 |

#### Discussion (linear)

The large 4096x4096-ish batch=1 matmuls show the most significant difference, with a ~48 ms median for TinyTorch versus ~0.9 ms for PyTorch. The reason is that this codebase uses a straightforward GEMM path.

### ReLU (8 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=1 C=64 HW=224×224 | 10.2211 | 2.1097 | 0.206 |
| N=2 C=128 HW=112×112 | 10.1764 | 1.9899 | 0.196 |
| N=4 C=256 HW=56×56 | 10.1956 | 1.9895 | 0.195 |
| N=8 C=512 HW=28×28 | 10.1818 | 1.9905 | 0.195 |
| N=16 C=1024 HW=14×14 | 10.1919 | 1.9895 | 0.195 |
| N=2 C=2048 HW=7×7 | 1.2236 | 0.3330 | 0.272 |
| N=8 C=32 HW=56×56 | 3.2289 | 0.6614 | 0.205 |
| N=32 C=128 HW=16×16 | 3.8751 | 0.7892 | 0.204 |

#### Discussion (ReLU)

For the five rows that sweep the ImageNet-style spatial pyramid at a roughly constant total element count (224^2 x 64 ≈ 112^2 x 128 ≈ ...), the median times for TinyTorch cluster near 10.2 ms while PyTorch remains near 2.0 ms. This indicates that the harness is stable and the bottleneck is not a result of a fluctuating timer. It represents a consistent factor-of-five disadvantage in elementwise fusion. The 7x7 high-channel row is less expensive in absolute terms because it simply contains fewer total elements.

### Global average pooling (6 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=1 C=64 HW=224×224 | 4.6886 | 0.2056 | 0.044 |
| N=4 C=256 HW=56×56 | 4.6828 | 0.2094 | 0.045 |
| N=16 C=512 HW=14×14 | 2.7130 | 0.1882 | 0.069 |
| N=32 C=128 HW=14×14 | 1.6056 | 0.1768 | 0.110 |
| N=16 C=64 HW=32×32 | 1.8767 | 0.1745 | 0.093 |
| N=8 C=512 HW=7×7 | 0.7515 | 0.1792 | 0.238 |

#### Discussion (global average pooling)

The results for global pooling follow a similar pattern to ReLU with a different constant factor. Reduction kernels in PyTorch are extremely efficient on this GPU for these sizes, while TinyTorch spends several milliseconds on large feature maps. The Torch/Tiny ratio improves on smaller spatial sizes, such as 7x7, where overhead becomes a more dominant factor in the total execution time.

As one man said ["Yaa To Win Hai Ya To Learn hai "](https://www.youtube.com/watch?v=6-KCe-Xqtgs).

### Fused cross-entropy (5 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=128 classes=10 | 0.0394 | 0.1549 | 3.934 |
| N=512 classes=16 | 0.0401 | 0.1550 | 3.862 |
| N=1024 classes=32 | 0.0400 | 0.1532 | 3.827 |
| N=4096 classes=48 | 0.0509 | 0.1536 | 3.017 |
| N=1024 classes=63 | 0.0406 | 0.1535 | 3.776 |

#### Discussion (loss)

In this section, the Torch/Tiny ratio is above 1, meaning TinyTorch’s fused kernel returned a lower median latency than torch.nn.functional.cross_entropy for this narrow class-count regime. This is consistent with a small specialized kernel that does not solve the general problem versus PyTorch’s general backward path, which handles more cases and more edge behavior. The 4096x48 row is still faster on TinyTorch but by a smaller factor because batch scaling starts to show. While this does not rescue a model whose time is dominated by convolution, as MNIST training still spends most of its execution in conv and GEMM kernels rather than the loss, it represents a real bright spot in the table.


# Testing and correctness

Each kernel and integration was tested prior to moving to the next implementation. We verified correctness by comparing results against PyTorch or by running small examples with analytically verified outcomes. Because these test cases were developed alongside the library and the codebase has evolved significantly across increments, many of them are no longer executable. They are included in the tests/ folder for the sake of completeness.


# Autograd

When a tensor with `requires_grad=1` participates in an operation, the functional code allocates an output tensor. If gradients are needed, it attaches a prev pointer to a Node struct that stores pointers to inputs, optional context for backward, and a generic function pointer to the correct backward implementation.

### Execution flow

Calling `optim.backward(Tensor *)` starts from the passed tensor and topologically sorts the graph by visiting dependencies. It then walks that list backward, calling each node’s generic backward function pointer  with the appropriate upstream gradient tensor. The implementation is located in `src/autograd/backward.cpp`. Cross-entropy is special-case: the fused loss writes directly into `logits->grad` in its backward pass rather than routing through a redundant dL/dL calculation, this is done for efficiency reasons.

### Memory management requirements

Because there is no automatic destructor graph like in PyTorch, Python examples must call `tensor_free(Tensor *)` for every allocated tensor once it leaves scope. Failure to do so will result in VRAM increasing until the process terminates. For this reason, training scripts in this repository free tensors aggressively after each step. 

This approach was directly inspired by Karpathy's micrograd. More information can be found here: [https://www.youtube.com/watch?v=VMj-3S1tku0&t=4726s](https://www.youtube.com/watch?v=VMj-3S1tku0&t=4726s)


# Planned Improvements
- [ ] Refactor the codebase to use cpp - features more.
- [ ] Mixed Precision.
- [ ] Use of tensor cores to speedup Convolution and Linear Layers.