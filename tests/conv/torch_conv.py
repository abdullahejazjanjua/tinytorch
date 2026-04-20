import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Configs: (N, Cin, Hin, Win, Cout, K, P_flag)
CONFIGS = [
    (3, 3, 5, 5, 2, 3, 1),
    (1, 3, 5, 5, 2, 3, 0),
    (8, 128, 32, 32, 32, 7, 1),
    (4, 64, 28, 28, 32, 5, 0),
]

def export_bin(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

def run():
    # Create data directory if it doesn't exist
    if not os.path.exists("data"): os.makedirs("data")

    for cfg in CONFIGS:
        n, ci, hi, wi, co, k, p_flag = cfg
        print(f"\n>>> TEST: N={n}, C={ci}, {hi}x{wi}, K={k}, P={'SAME' if p_flag==1 else 'VALID'}")
        
        # 1. Prepare PyTorch Data
        x = torch.randn(n, ci, hi, wi, requires_grad=True, device='cuda')
        w = torch.randn(co, ci, k, k, requires_grad=True, device='cuda')
        
        # 2. PyTorch Reference
        p_val = (k - 1) // 2 if p_flag == 1 else 0
        out = F.conv2d(x, w, padding=p_val, stride=1)
        dout = torch.randn_like(out)
        out.backward(dout)
        
        # 3. Export to binary for C
        export_bin(x, 'data/input.bin')
        export_bin(w, 'data/filters.bin')
        export_bin(out, 'data/out_ref.bin')
        export_bin(dout, 'data/dout.bin')
        export_bin(w.grad, 'data/gw_ref.bin')
        export_bin(x.grad, 'data/gx_ref.bin')
        
        # 4. Invoke C Binary in previous folder
        cmd = ["./a.out", str(n), str(ci), str(hi), str(wi), str(co), str(k), str(p_flag)]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(res.stdout)
            else:
                print(f"STDOUT: {res.stdout}")
                print(f"STDERR: {res.stderr}")
                break
        except Exception as e:
            print(f"Subprocess Error: {e}")
            break

if __name__ == "__main__":
    run()