# TinyTorch

TinyTorch is a teaching-scale deep learning backend: enough moving parts to train a small convolutional network on MNIST, but small enough that you can still read the CUDA, the autograd, and the Python bindings without a map the size of a poster. The stack includes a `Tensor` type (FP32, optional `requires_grad`, can live on host or device), a reverse-mode autograd that walks a tape built from C++ functionals, hand-written kernels for 2D convolution (forward plus gradients w.r.t. input and weights), dense linear layers, ReLU, global average pooling, and a fused softmax-with-cross-entropy loss. Optimizer support is plain SGD. Everything that touches training math on the GPU is meant to be inspectable; nothing here tries to compete feature-for-feature with PyTorch.

The normal way to *use* TinyTorch from userland is Python: CMake builds several PyBind11 extension modules (`base`, `nn`, `optim`, `mnist_io`) that sit next to `train_mnist.py`. The heavy lifting still lives in C++/CUDA under `include/`, `src/`, and `cuda/`.

This README walks through what you need on your machine, how to compile, how the data layout works, how to call the API with random tensors, what the CUDA kernels actually assume (so you do not hit opaque launch errors), and how the project behaved in a documented MNIST run and a full layer-wise benchmark pass against PyTorch on one workstation.

---

## Repository layout (where to look)

- **`bind/`** — CMake project that builds `dl_core` and the Python modules. The `CMakeLists.txt` here globs `src/**/*.cpp` and `cuda/*.cu`, links CUDA runtime and CURAND, and emits `.so` / `.pyd` into the **parent** directory (`tinytorch/`), not into `build/`, so imports work when your shell’s cwd is the repo root.
- **`include/`** — Public C headers for tensors, autograd `Node` wiring, and declarations for kernels (conv, matmul, pooling, softmax–CE, ReLU, etc.).
- **`src/`** — Autograd driver (`src/autograd/backward.cpp`), functional wrappers that allocate outputs and attach `prev` nodes, NN layer constructors, and optimizers.
- **`cuda/`** — Device code: convolution, GEMM-related pieces, pooling, fused loss, utilities.
- **`mnist-dataloader/`** — Minimal IDX reader used by `mnist_io` and by the C++ tests that need real bytes.
- **`tests/`** — C++ binaries for gradient checks, conv wrappers, autograd stress tests, and historical “dump tensors and diff against PyTorch” flows.

If you are debugging a bad gradient, start in `src/functional/` for the autograd link, then drop into the matching `.cu` file. If Python says it cannot import `nn`, you are usually in the wrong working directory or the build never produced the extensions next to the scripts.

---

## Prerequisites

You will need a working **NVIDIA GPU stack**: a driver and a **CUDA toolkit** that match closely enough that `nvcc` can compile for your card. TinyTorch does not implement a serious CPU fallback for training; the point is to run kernels on the device.

**CMake 3.18 or newer** is required because the project uses modern CMake patterns and pulls in `CUDAToolkit` through `find_package`. Older distros sometimes ship CMake 3.10; upgrade before complaining that `CUDA_ARCHITECTURES` does nothing.

**Python 3** must have **pybind11** available. The CMake file runs `python3 -c "import pybind11; print(pybind11.get_cmake_dir())"` to locate PyBind’s CMake package. Installing pybind11 in the same interpreter you use at build time (`pip install pybind11`) is usually enough. If CMake says it cannot find pybind11, either the wrong Python is on `PATH` or the package is missing in that environment.

**Compiler**: on Linux that is typically GCC or Clang with C++14 support; on Windows, MSVC must cooperate with NVCC and your CUDA version (check NVIDIA’s compatibility matrix). Mixed toolchains are the most common source of “clean on Linux, broken on Windows” stories.

---

## GPU architecture flags (important)

CUDA kernels are compiled for specific **streaming multiprocessor versions**. In `bind/CMakeLists.txt` you will find:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
```

That bakes in **sm_75**, **sm_80**, and **sm_86**. Cards in those families run out of the box. If your GPU is different—**sm_89 (Ada)**, **sm_90 (Hopper)**, a **notebook 50-series**, etc.—you **must** add the right architecture number (or replace the list) and rebuild. If you skip this step, typical failure modes are runtime errors like *no kernel image is available for execution on the device* even though the binary linked fine.

There is no universal “compile for everything” flag that stays fast and small; pick the arch you actually run on.

---

## Building from source

```text
cd bind
mkdir build
cd build
cmake ..
cmake --build .
```

On Windows with multi-config generators you still get a `Release` folder under `build/` for object files, but the **Python extension modules** are configured with `LIBRARY_OUTPUT_DIRECTORY` pointing at **`../`** relative to `bind/`, i.e. the directory that contains `train_mnist.py`. After a successful build you should see shared libraries whose names depend on the platform (`base*.so`, `nn*.so`, … on Linux, or `.pyd` on Windows).

What gets built:

1. **`dl_core`** — One shared library with tensor allocation, all the `.cpp` functionals and NN glue, and all `.cu` translation units linked with `CUDA::cudart` and `CUDA::curand`.
2. **`base`** — PyBind module: create/free tensors, move host↔device, expose `data` as a NumPy-compatible buffer and `shape` as a Python list.
3. **`nn`** — PyBind module: layer classes (`Conv2D`, `Linear`, `ReLU`, `GlobalPooling`, `CrossEntropy`) and their `forward` methods.
4. **`optim`** — PyBind module: `SGD` plus a module-level **`backward(tensor)`** that runs the autograd pass from a scalar (or in CE’s case, from the loss tensor’s backward hook into logits).
5. **`mnist_io`** — PyBind module for loading IDX MNIST into RAM and filling batch tensors.

If `import base` fails in Python, check: (1) cwd or `PYTHONPATH` includes the folder with the `.so`, (2) `dl_core` is visible to the loader (same directory or `LD_LIBRARY_PATH` / `PATH` on Windows), (3) you did not build Debug on a machine that only has Release runtimes, etc.

---

## MNIST data (IDX format)

The training script `train_mnist.py` expects **raw MNIST IDX** files, not PNG folders and not necessarily torchvision’s cache layout. The usual layout is a `data/` directory next to the scripts with names along the lines of:

- `train-images.idx3-ubyte` — training images  
- `train-labels.idx1-ubyte` — training labels  
- `t10k-images.idx3-ubyte` — test images (optional for evaluation)  
- `t10k-labels.idx1-ubyte` — test labels  

You can pass `--data-dir` to point somewhere else. The loader reads the whole set into RAM once (`load_dataset_in_ram`), then each epoch shuffles index lists and copies batches into tensors with `load_batch_to_tensor`. There is **no** automatic download in this README: grab the official IDX files from the usual MNIST hosting mirrors and unpack them yourself.

The log line *Warning: truncating to full batches: 59968 samples* means the raw dataset size does not divide evenly by the batch size you chose (here **64**), so the last partial batch is dropped and each epoch uses **59 968** training samples instead of 60 000. Both TinyTorch and the PyTorch comparison below used the same convention.

---

## Python API / examples with random data

The following examples avoid MNIST entirely: they allocate tensors, fill them with NumPy RNG output, run forward and backward, and print a scalar loss. They assume the extension modules are importable from your current working directory (build the project first).

### Design note: why create on CPU then `tensor_to_gpu`

`tensor_create(shape, requires_grad, on_gpu)` can create tensors directly on the GPU (`on_gpu=1`). In that case the PyBind-exposed `data` buffer is still a view over device memory; writing from NumPy across PCIe in a portable way is awkward. The examples below create tensors **on the CPU** (`on_gpu=0`), assign `x.data[:] = ...` from NumPy, then call `base.tensor_to_gpu(x)`. That pattern matches how the older `test.py` in this repo fills buffers before calling layers. Layers like `Conv2D` and `Linear` still allocate their **weights on the GPU** inside the C constructors.

### Example 1 — Linear, fused cross-entropy, backward

```python
import numpy as np
import base
import nn
import optim as optim_mod

N, in_features, num_classes = 16, 64, 10  # fused CE requires num_classes < 64

# Input activations: [N, in_features], requires_grad=1, start on CPU then upload
x = base.tensor_create([N, in_features], 1, 0)
x.data[:] = np.random.standard_normal(N * in_features, dtype=np.float32)
base.tensor_to_gpu(x)

# Linear(in, out, has_bias, requires_grad). Weights are (in_features, out_features) in memory
# and are initialized on the GPU by the C++ constructor.
lin = nn.Linear(in_features, num_classes, 1, 1)
logits = lin.forward(x)

# Labels: length N, integer class id stored as float in device memory (what the fused CE kernel reads)
labels = base.tensor_create([N], 0, 0)
labels.data[:] = np.random.randint(0, num_classes, size=N).astype(np.float32)
base.tensor_to_gpu(labels)

ce = nn.CrossEntropy(1)
loss = ce.forward(logits, labels)

# Reverse-mode: starts from loss, walks the graph, fills .grad on logits and below
optim_mod.backward(loss)

base.tensor_to_cpu(loss)
print("loss", float(loss.data[0]))

base.tensor_free(loss)
base.tensor_free(logits)
base.tensor_free(x)
base.tensor_free(labels)
```

After `backward`, `lin.weights.grad` and `lin.bias.grad` hold GPU tensors (same shapes as the parameters) that you can read back with `tensor_to_cpu` if you want to inspect numbers.

To actually **train**, construct `optim.SGD([lin.weights, lin.bias, ...], lr)`, call `zero_grad()` on the optimizer (which zeroes each parameter’s `.grad`), then `backward(loss)`, then `step()`, same as in `train_mnist.py`.

### Example 2 — Convolution, global pool, linear, cross-entropy

This is a miniature CNN-shaped graph. The convolution constructor is `Conv2D(in_channels, out_channels, kernel_size, padding_flag, requires_grad)`. Here **`padding_flag=1`** means “same” padding in the sense used in this codebase (output spatial size matches input for odd kernels with the usual half-size padding); **`0`** means valid (no padding). Pick shapes that satisfy the **grid limits** spelled out later in this README; the numbers below are safe on typical NVIDIA parts.

```python
import numpy as np
import base
import nn
import optim as optim_mod

N, cin, cout, h, w, k = 4, 8, 16, 32, 32, 3
num_classes = 10

x = base.tensor_create([N, cin, h, w], 1, 0)
x.data[:] = np.random.standard_normal(N * cin * h * w, dtype=np.float32)
base.tensor_to_gpu(x)

conv = nn.Conv2D(cin, cout, k, 1, 1)
y = conv.forward(x)

pool = nn.GlobalPooling(1)
z = pool.forward(y)

lin = nn.Linear(cout, num_classes, 1, 1)
logits = lin.forward(z)

labels = base.tensor_create([N], 0, 0)
labels.data[:] = np.random.randint(0, num_classes, size=N).astype(np.float32)
base.tensor_to_gpu(labels)

ce = nn.CrossEntropy(1)
loss = ce.forward(logits, labels)

optim_mod.backward(loss)

base.tensor_to_cpu(loss)
print("loss", float(loss.data[0]))

base.tensor_free(loss)
base.tensor_free(logits)
base.tensor_free(z)
base.tensor_free(y)
base.tensor_free(x)
base.tensor_free(labels)
```

You are responsible for freeing every tensor you no longer need; there is no Python GC for CUDA allocations like in PyTorch. Training loops in this project free activations after each step to avoid leaking VRAM.

---

## Hardware and API limitations (convolution in depth)

The convolution implementation in `cuda/conv.cu` is written for **training small CNNs**, not for general computer vision at arbitrary batch width and channel width. Limits come from **template choices**, **shared memory tiling**, and **CUDA’s maximum grid dimensions**.

### Kernel sizes, stride, padding, and “what kind of conv”

- **Backward w.r.t. weights** is instantiated only for **square odd** kernels **1, 3, 5, and 7**. Other sizes either will not compile the right template or will hit a `default` branch that does not launch a kernel. Plan your architecture around those kernels if you need training gradients.
- **Stride** is **1** and **dilation** is **1**. The indexing and shared-memory tile logic does not implement dilated or strided convolution as first-class citizens.
- There is **one spatial conv**: no **groups**, no **depthwise separable**, no **transposed** convolution.
- **Padding** is either **“same”** in the sense that the output height and width match the input (implemented by padding **(K−1)/2** on each side when the padding flag is on), or **valid** (zero padding). For **valid** convolution you need the input spatial size large enough that **H_out = H − K + 1** (and similarly for **W**) is positive—i.e. generally **H, W ≥ K**.

### Why you sometimes see `invalid configuration argument`

On NVIDIA GPUs, **`gridDim.z`** is limited (historically to **65,535** for many chips). TinyTorch maps different phases of convolution onto 3D grids. In practice three products must stay within that cap:

1. **Backward w.r.t. input:** the **Z** dimension is proportional to **N × Cin** (batch × input channels). You need **N × Cin ≤ 65,535**.
2. **Backward w.r.t. weights:** the **Z** dimension is **Cin × Cout**. You need **Cin × Cout ≤ 65,535**.  
   This is **independent** of the batch size **N**. A “square” block like 512 input channels and 512 output channels is **over 262,000** in that product and will fail even if **N** is 1. The README’s benchmarks intentionally cap channel products for this reason.
3. **Forward** pass: the **Z** grid is built from **ceil(N/2) × ceil(Cout/2)** (see the `COARSE_FACTOR` in `conv.cu`). That product must also stay **≤ 65,535** on typical hardware. Very large batch together with very wide output channel counts can hit this even when the two backward constraints look fine.

If any of these are violated, CUDA returns an error at launch time—often surfaced as *invalid configuration argument* at the line that checks `cudaGetLastError()` after the kernel.

### Occupancy and numerics

The backward-weight kernel uses a **fixed block size of 16** and keeps a small **on-chip array** sized for **K × K** filters. That is a deliberate trade to control register pressure; it is not tuning for every GPU generation.

All training here is in **FP32** with “do the obvious multiply-add loop” semantics. Compared to **cuDNN**, which may reorder reductions and use different accumulation strategies, you can see relative errors on the order of **10⁻³** on unfriendly shapes (many input channels, **K = 7**, etc.). That is expected for different floating-point trees, not necessarily a bug.

### Global average pooling

Earlier versions of the global pooling kernel ran into grid limits when **batch × channels** grew large, similar in spirit to the conv backward-input issue. The current pooling path was reworked so that large **N × C** products are handled without leaning on an oversized **`gridDim.z`**. Pooling is still meant for **reasonably sized feature maps** in small CNNs; it is not a general ND reduction framework.

### Fused softmax + cross-entropy

The fused loss kernel is only valid when **`num_classes` is strictly less than 64** (practically you use **1 through 63** output logits). The implementation assumes a small enough class dimension that a block can cooperate over the row in the way the CUDA file lays out. It also **recomputes** quantities in the backward path instead of stashing extra fields on `Tensor` to remember forward state—less surface area in the struct, a bit more redundant math on the backward pass.

If you need **ImageNet-scale** 1000-way classifiers without touching PyTorch, you would need a different loss path.

### “Everything else”

**Linear** and **ReLU** do not publish the same grid-algebra essay, but you are still limited by **VRAM**, **Python overhead**, and the lack of fused epilogue kernels that a production library stacks together. Out-of-memory is still out-of-memory.

---

## End-to-end MNIST training vs PyTorch (matched recipe)

The following numbers come from one pair of runs on the **same machine**, **same data root**, **same architecture description**, and **same optimization hyperparameters**, differing only in whether the loop was driven by TinyTorch+IDX or by PyTorch+`torchvision` MNIST tensors.

**Setup**

- **Training samples per epoch:** 59 968 (full batches only; remainder of 60 000 dropped for batch size 64).
- **Batch size:** 64.
- **Learning rate:** 0.05.
- **Epochs:** 10.
- **Architecture (both sides):** convolutional stem **1 → 48 → 96** channels with **5×5** convs and padding chosen to match the TinyTorch convention, **global average pooling** to a vector of width 96, then MLP head **96 → 128 → 10** with ReLU between (exact layer order matches the twin scripts in this repo).
- **TinyTorch** reads IDX from `.../data`; **PyTorch** used MNIST rooted at the same path.
- **Test evaluation** in the logs below uses **9 984** test samples (again full batches only from the 10 000 test images).

**Summary table**

| | **TinyTorch** | **PyTorch (CUDA)** |
|---|----------------|---------------------|
| **Total training wall time** | **1261 s** (~21 min) | **64 s** |
| **Mean train loss (epoch 10)** | **0.183** | **0.308** |
| **Train accuracy (epoch 10)** | **94.5%** | **91.0%** |
| **Test: mean loss** | **0.143** | **0.205** |
| **Test: accuracy** | **95.6%** | **94.2%** |

**How to read it**

Both runs show loss decreasing epoch over epoch and accuracy increasing; neither training curve is “broken.” TinyTorch finishes an epoch in roughly **two minutes** early on and stabilizes near **~126 s** per epoch later; PyTorch finishes most epochs in **five to seven seconds** on the same GPU. The ~**20×** total-time ratio is dominated by **custom kernels without cuDNN-level fusion**, **explicit host/device staging** in the Python training loop, and **allocator behavior**, not because TinyTorch magically does more floating-point work per epoch—the batch count is fixed.

The **final train loss** is **lower** on TinyTorch and **train accuracy** is **higher**, while **test accuracy** is also **slightly higher** on TinyTorch on this split. These are **single seeds / single runs** without exhaustive sweeps; differences can come from **initialization details**, **numerical ordering**, **data order** (even with shuffle, the stacks are not bitwise identical), and **subtle BN-free CNN differences** between the twins. The point for this README is not that TinyTorch is a better learner, but that it **does learn** and lands in a ballpark that PyTorch also reaches, while being **much slower per wall-clock epoch**.

---

## Layer micro-benchmarks (full table and interpretation)

This section records a **single** micro-benchmark session so you can see raw medians, not cherry-picked anecdotes.

### Environment

| Field | Value |
|-------|--------|
| **GPU** | NVIDIA RTX A2000 12GB |
| **PyTorch** | 2.6.0+cu124 |
| **CUDA** | 12.4 |
| **Report timestamp** | 2026-05-02T04:34:17 |

### Timing protocol

For each row below, the harness measures **one forward pass and one backward pass** through the relevant operator (with a synthetic upstream gradient where the output is not a scalar—exactly like training would eventually propagate a gradient into that tensor). Times are **median milliseconds** over **80** timed iterations after **20** warmup iterations. Each timed iteration is wrapped with **CUDA device synchronization** so asynchronous launch does not lie about which pass finished.

PyTorch’s side had **TensorFloat-32 disabled** for both **cuBLAS matmul** and **cuDNN** so the comparison stays in an **FP32** spirit similar to the TinyTorch kernels.

The column **Torch/Tiny** is the ratio **(PyTorch median ms) / (TinyTorch median ms)**. If it is **less than 1**, PyTorch was **faster** on that row (it took fewer milliseconds). If it is **greater than 1**, TinyTorch was **faster**.

Between **layer families** (e.g. after finishing all convolution rows and before starting linear rows), the harness also **synchronized**, **cleared the CUDA allocator cache**, and **summed a ~96 MiB scratch tensor** once. That is not a feature of normal training; it is there so the **next** family’s cold-cache behavior is a bit less contaminated by the previous family’s L2 state. **Individual rows** within a family were **not** separated by that flush; only broad section boundaries were.

**All 49 configurations launched successfully** (zero skipped rows). Each configuration respected the convolution grid constraints documented earlier in this README.

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

**Discussion (convolution).** On almost every line, PyTorch+cuDNN finishes the fused forward-backward pair in a fraction of the milliseconds TinyTorch needs. The ratio is least punishing on **tiny** problems (single-channel MNIST-ish maps, few output channels) where fixed launch and synchronization overhead is a measurable slice of TinyTorch’s time—see the row with **Ci=1**, **Torch/Tiny**, ~**0.97**. As soon as **width × height × channels** grows, TinyTorch times climb into **tens to hundreds of milliseconds** per step while PyTorch stays in **single digits** for comparable geometry. The **224×224** stem-style rows with **k = 7** show that spatial size still hurts the teaching kernel a lot more than it hurts a vendor library that has spent years on occupancy and memory traffic. None of this is surprising; it is the cost of not shipping a full autotuning conv factory.

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

**Discussion (linear).** The big **4096×4096**-ish **batch=1** matmuls are the starkest: **~48 ms** median vs **~0.9 ms**. That is cuBLAS doing what cuBLAS does—tile sizes, warp-level primitives, possibly better pipelining—versus a straightforward GEMM path in this codebase. As the **batch dimension** grows but the **GEMM footprint** shrinks toward “MNIST-head-sized” layers, **Torch/Tiny** creeps toward **0.3–0.4**, meaning PyTorch is still ahead but not by two orders of magnitude anymore because **fixed costs** eat a larger share on both sides.

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

**Discussion (ReLU).** For the five rows that sweep the **ImageNet-ish spatial pyramid** at **roughly constant total element count** (**224² × 64** ≈ **112² × 128** ≈ …), TinyTorch’s median times **cluster near ~10.2 ms** while PyTorch sits near **~2.0 ms**. That tells you the harness is stable and the bottleneck is not a flaky timer—it is a consistent factor-of-five-ish disadvantage on elementwise fusion. The **7×7** / high-channel row is cheaper in absolute terms because there are simply fewer elements.

### Global average pooling (6 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=1 C=64 HW=224×224 | 4.6886 | 0.2056 | 0.044 |
| N=4 C=256 HW=56×56 | 4.6828 | 0.2094 | 0.045 |
| N=16 C=512 HW=14×14 | 2.7130 | 0.1882 | 0.069 |
| N=32 C=128 HW=14×14 | 1.6056 | 0.1768 | 0.110 |
| N=16 C=64 HW=32×32 | 1.8767 | 0.1745 | 0.093 |
| N=8 C=512 HW=7×7 | 0.7515 | 0.1792 | 0.238 |

**Discussion (global pool).** Same story as ReLU at a different constant factor: reduction kernels in PyTorch are **extremely cheap** on this GPU for these sizes, while TinyTorch spends **a few milliseconds** on large feature maps. The **Torch/Tiny** ratio **improves** (gets closer to 1, though still below) on **smaller spatial sizes** like **7×7**, where overhead is relatively louder.

### Fused cross-entropy (5 configurations)

| Configuration | Tiny ms | Torch ms | Torch/Tiny |
|---------------|---------|----------|------------|
| N=128 classes=10 | 0.0394 | 0.1549 | 3.934 |
| N=512 classes=16 | 0.0401 | 0.1550 | 3.862 |
| N=1024 classes=32 | 0.0400 | 0.1532 | 3.827 |
| N=4096 classes=48 | 0.0509 | 0.1536 | 3.017 |
| N=1024 classes=63 | 0.0406 | 0.1535 | 3.776 |

**Discussion (loss).** Here **Torch/Tiny** is **above 1**, meaning **TinyTorch’s fused kernel returned a lower median latency** than `torch.nn.functional.cross_entropy` for this narrow class-count regime. That is consistent with a **small specialized kernel** that does not solve the general problem versus PyTorch’s **general backward** (which handles more cases and more edge behavior). The **4096×48** row is still faster on TinyTorch but by a smaller factor because **batch scaling** starts to show. None of this rescues a model whose time is dominated by convolution—MNIST training still spends its life in conv+GEMM, not in the loss—but it is a real bright spot in the table.

---

## Correctness and regression tests (C++)

Before you trust a new kernel, walk the **tests/** tree:

- **`tests/autograd/`** — Builds that drive the autograd engine on hand-constructed graphs and compare against reference gradients.
- **`tests/wrapper/`** — Per-op executables that link subsets of the stack for conv, linear, ReLU, CE, etc.
- **`tests/model-definition/`** — Historical flow that runs a forward+backward story close to a tiny CNN and compares tensor dumps against PyTorch with tight **max-relative-error** thresholds on logits and intermediate gradients.

Those binaries are not run automatically by this README; you compile them with appropriate `nvcc` / `g++` invocations mirroring the `.cpp` and `.cu` lists in your own environment (older comments in this repo showed one long `nvcc` command line for a combined model test). They are still the right place to look when someone says “I changed the conv backward and now MNIST diverges.”

---

## Autograd (how backward actually runs)

When a tensor with `requires_grad` participates in an operation, the functional code allocates an output tensor and, if gradients are needed, attaches a **`prev`** pointer to a **`Node`** struct that stores pointers to inputs, optional context for backward, and a function pointer to the right backward implementation.

Calling **`optim.backward(loss)`** starts from the loss tensor, **topologically sorts** the graph by visiting dependencies, then walks that list backward calling each node’s backward hook with the appropriate upstream gradient tensor (the implementation lives in `src/autograd/backward.cpp`). Cross-entropy is special-cased: the fused loss writes directly into **`logits->grad`** in its backward rather than routing through a meaningless `dL/dL`.

Because there is **no** automatic destructor graph like PyTorch’s, **Python examples must `tensor_free` everything** they allocate once tensors leave scope, or VRAM will walk upward until the process dies. Training scripts in this repo free tensors aggressively after each step for that reason.

---

If something in this README disagrees with the code, trust the compiler and the kernels first, then open an issue or fix the docs.