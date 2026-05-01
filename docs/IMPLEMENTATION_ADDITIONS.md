# TinyTorch — implementation additions (branch `mumtaz`)

This document describes work added on top of the existing TinyTorch codebase: new operators, optimizer, GEMM extension, build system, tests, and verification notes. It is aimed at anyone reviewing or continuing the project.

---

## 1. Summary of features added

| Feature | Location | Role |
|--------|----------|------|
| **ReLU** | `cuda/relu.cu`, `src/functional/relu.cpp`, `src/nn/relu.cpp` | Elementwise `max(0, x)` forward; backward masks gradient by input sign. Wired into autograd like other layers. |
| **GEMM (matmul + optional bias)** | `cuda/mm.cu`, `src/functional/linear.cpp`, `src/nn/linear.cpp` | `matmul_forward_pass(A, B, bias_or_null, C)` fuses bias add when `bias != nullptr`. `matmul_backward_pass_bias` computes `db = sum(dC, axis=0)`. Linear layer allocates bias when `has_bias == 1`. |
| **SGD** | `cuda/sgd.cu`, `include/optim.h`, `src/optim/sgd.cpp` | CUDA kernel: `param[i] -= lr * grad[i]`. C++ wrapper `SGD` holds `std::vector<Tensor*>`, exposes `step()` and `zero_grad()`. |

---

## 2. API and signature changes

### 2.1 `include/tensor.h`

- **ReLU**
  - `void relu_forward_pass(const Tensor *input, Tensor *output);`
  - `void relu_backward_pass(const Tensor *input, const Tensor *dout, Tensor *grad_in);`

- **Matmul / GEMM**
  - `void matmul_forward_pass(const Tensor *A, const Tensor *B, const Tensor *bias, Tensor *C);`  
    Pass `bias == nullptr` for plain matmul.
  - `void matmul_backward_pass_bias(const Tensor *dC, Tensor *db);`  
    `db` shape `[N]` when `dC` is `[M, N]`.

### 2.2 `include/functional.h`

- `Tensor* linear_functional_forward(Tensor *input, Tensor *weights, Tensor *bias, int requires_grad);`  
  `bias` may be `nullptr`.

- ReLU functional symbols: `relu_functional_forward`, `relu_functional_backward`.

### 2.3 `include/nn.h`

- `class ReLU` — mirrors `GlobalPooling` style (`requires_grad`, `forward`).

- **`Linear` constructor signature changed**  
  `Linear(int in_features, int out_features, int has_bias, int requires_grad);`  
  - `has_bias == 0` → `bias` is `nullptr`, behavior matches old matmul-only linear.
  - `has_bias == 1` → `bias` allocated on GPU, zero-initialized (`cudaMemset`).

### 2.4 `include/optim.h`

- Extern `"C"` : `void sgd_step_pass(Tensor *param, float lr);` (requires `param->grad` valid on GPU.)

- Class `SGD`:
  - `SGD(std::vector<Tensor*> params, float lr);`
  - `void step();` — skips entries with missing `grad` or `requires_grad == 0`.
  - `void zero_grad();` — `cudaMemset` each `grad->data`.

---

## 3. Autograd wiring

- **ReLU** — single input node; backward uses stored input to gate `dout` (no separate mask tensor).
- **Linear with bias** — when bias present, `Node` has `num_inputs == 3`: `[input, weights, bias]`. Backward calls `matmul_backward_pass_A`, `matmul_backward_pass_B`, and `matmul_backward_pass_bias` as appropriate.

**SGD is not part of autograd.** It updates parameters after `backward(loss)` according to gradients already accumulated in each tensor’s `grad` field.

---

## 4. Build system (`Makefile`)

- Variable `ARCH` defaults to `-arch=sm_75` (matches original README assumptions).
- On **Ampere** GPUs (e.g. RTX A2000), set `ARCH := -arch=sm_86` (or patch with `sed` on the machine).
- Useful targets:
  - `make all-tier1` — all wrapper tests + both autograd test binaries + integration tests (`full-pipeline-test`, `full-pipeline-sgd-test`).
  - `make run-tier1` — build tier-1, then run each binary sequentially.
  - `make build/test_matmul`, `make build/test_conv` — Tier-2 PyTorch-comparison harnesses.

Link flag: `-lcurand` (Xavier init, etc.)

---

## 5. Tests added or revived

### 5.1 New wrapper-style tests (`tests/wrapper/`)

| File | What it checks |
|------|----------------|
| `relu-wrapper.cpp` | ReLU forward, graph wiring, backward mask. |
| `linear-bias-wrapper.cpp` | `Linear(..., has_bias=1, ...)`: forward `X @ W + b`, `dW`, `dX`, `db = sum(dY, axis=0)`. |
| `sgd-wrapper.cpp` | `step()` arithmetic, `zero_grad()`, multi-tensor iteration. |

### 5.2 Existing tests updated

- `linear-wrapper.cpp`, `autograd-test.cpp`, `exhaustive-autograd-test.cpp` — `Linear` constructor gains `has_bias`; these pass `0` to preserve prior numeric expectations.

### 5.3 `tests/model-definition/model-def.cpp`

- `matmul_forward_pass(..., nullptr, ...)`.
- Fix: `matmul_backward_pass_B` called with correct 4 arguments (historical typo removed).

### 5.4 Tier-2 C harnesses (`tests/matmul/test_matmul.c`, `tests/conv/test_conv.c`)

- Updated for current `tensor_create` arity, GPU residency, separate `matmul_backward_pass_A` / `_B`, and **`matmul_forward_pass` bias argument**.
- **C++/nvcc portability:** compound literal shapes `(int[]){ … }` were replaced by named stack `int shape_…[] = { … }` so **g++ (host side of nvcc)** does not error with “taking address of temporary array”.

Companion Python scripts (`torch_matmul.py`, `torch_conv.py`) generate reference `.bin` files. If PyTorch is built against a CUDA version newer than the host driver, use **`device='cpu'`** in the conv generator (or/install a Torch build matched to the driver). Matmul reference generation already runs comfortably on CPU.

---

## 6. Verification status (conceptual)

- **Tier-1 (`make run-tier1`)** — exercises forward + backward for conv, pooling, softmax–CE (fused), matmul/GEMM (with and without bias in dedicated tests), ReLU, and SGD step logic; chained autograd (linear+CE, conv+pool+linear+CE); **full-pipeline-test** (`Conv→ReLU→GlobalPool→Linear(bias)→CE`, single `backward(loss)`); **full-pipeline-sgd-test** (same graph, then **real grads** + **one SGD step** + **zero_grad** on conv/linear/bias parameters).

- **Tier-2** — `build/test_matmul` + `tests/matmul/torch_matmul.py` and `build/test_conv` + `tests/conv/torch_conv.py` compare kernels to PyTorch on sampled shapes (when run successfully on a GPU machine with generated data).

- **Softmax–CE fused kernel** retains documented constraints (e.g. class-count assumptions in warp layout — see README).

- **`model-def` + MNIST + `compare.py`** — remains a separate validation path needing IDX files under `../../data/`.

Correctness means “matches tests and documented scope,” not formal proof for all tensors and strides.

---

## 7. SGD integration note

**No production training loop calls SGD yet.** The only wired usage today is `tests/wrapper/sgd-wrapper.cpp`. To train end-to-end, after computing `backward(loss)`, build a vector of pointers to trainable tensors (weights, biases, etc.) and call:

```cpp
#include "optim.h"
SGD opt({ ... }, lr);
backward(loss);   /* or equivalent manual backward */
opt.step();
opt.zero_grad();
```

Ordering relative to clearing gradients (`zero_grad` before next forward vs after `step`) should match whatever convention you adopt (PyTorch clears before forward; TinyTorch wrappers assume gradients accumulate unless zeroed explicitly).

---

## 8. Kernel implementation style (what was intentionally “simple”)

New CUDA entry points (ReLU, bias-gradient reduction for matmul, SGD step) are **straightforward, correct-first** implementations:

- Elementwise ReLU and SGD: one thread per element; adequate for bandwidth-bound work; vectorization (`float4`) would be incremental improvement.

- `matmul_backward_bias_kernel`: column-wise sum via one thread reducing over `M` — simple and correct; a tiled / warp-reduction version would scale better for very large batch dimension `M`.

Existing matmul tiling and Abdullah’s conv / pooling / softmax kernels retain their original optimization style.

---

## 9. Git workflow reference

Feature work landed on branch **`mumtaz`** and was pushed to the shared remote; small commits grouped by logical units (kernel, functional, nn class, matmul bias pieces, Makefile, tests, portability fix for Tier-2 C files).

---

## 10. File inventory (high level)

**New:**

- `cuda/relu.cu`, `cuda/sgd.cu`
- `src/functional/relu.cpp`, `src/nn/relu.cpp`
- `src/optim/sgd.cpp`
- `include/optim.h`
- `tests/wrapper/relu-wrapper.cpp`, `linear-bias-wrapper.cpp`, `sgd-wrapper.cpp`
- `tests/integration/full-pipeline-sgd-test.cpp` — same graph as above; after `backward(loss)`, one `SGD::step()` on conv/linear weights + bias verifies `new = old - lr * grad`, then `zero_grad()` clears grads.
- `Makefile`
- This doc: `docs/IMPLEMENTATION_ADDITIONS.md`

**Heavily edited:**

- `cuda/mm.cu`, `include/tensor.h`, `include/functional.h`, `include/nn.h`
- `src/functional/linear.cpp`, `src/nn/linear.cpp`
- Selected tests under `tests/autograd/`, `tests/wrapper/`, `tests/model-definition/`, `tests/matmul/`, `tests/conv/`

---

*End of implementation notes.*
