import torch
import numpy as np
import os

def export_bin(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

CONFIGS = [
    (4, 4, 4),
    (8, 16, 8),
    (32, 32, 32),
    (64, 64, 64),
]

def run():
    os.makedirs("data", exist_ok=True)

    for M, K, N in CONFIGS:
        print(f"\nTEST: A[{M},{K}] @ B[{K},{N}]")

        # fresh tensors per config
        A = torch.randn(M, K, requires_grad=True)
        B = torch.randn(K, N, requires_grad=True)

        # forward reference
        C = A @ B

        # upstream gradient
        dC = torch.randn_like(C)

        # backward reference
        C.backward(dC)

        # ---- IMPORTANT: per-config file isolation ----
        export_bin(A, f"data/A_{M}_{K}_{N}.bin")
        export_bin(B, f"data/B_{M}_{K}_{N}.bin")

        export_bin(C, f"data/C_ref_{M}_{K}_{N}.bin")
        export_bin(dC, f"data/dC_{M}_{K}_{N}.bin")
        export_bin(A.grad, f"data/dA_ref_{M}_{K}_{N}.bin")
        export_bin(B.grad, f"data/dB_ref_{M}_{K}_{N}.bin")

        print("exported data for this config")

if __name__ == "__main__":
    run()
