import torch
import torch.nn.functional as F
import numpy as np

def load_bin(name, shape):
    return torch.from_numpy(np.fromfile(f"/content/{name}", dtype=np.float32).copy()).view(*shape)

def compare(name, custom, torch_val, eps=1e-8):
    custom, torch_val = custom.float(), torch_val.float()
    abs_diff = torch.abs(custom - torch_val)
    denom = torch.max(torch.abs(custom), torch.abs(torch_val)) + eps
    rel_diff = abs_diff / denom
    max_rel = rel_diff.max().item()
    print(f"[{name:.<30}] Max Rel Diff: {max_rel:.2e} | {'PASS ✅' if max_rel < 1e-3 else 'FAIL ❌'}")

cpp_in = load_bin("input.bin", (4, 1, 28, 28))
cpp_w  = load_bin("weights.bin", (8, 1, 3, 3))
cpp_fc_w = load_bin("fc_weights.bin", (8, 10))
cpp_l  = load_bin("labels.bin", (4,)).long()

cpp_fwd = load_bin("fwd_logits.bin", (4, 10))
cpp_g_logits = load_bin("grad_logits.bin", (4, 10))
cpp_g_fc_w = load_bin("grad_fc_weights.bin", (8, 10))
cpp_g_pooled = load_bin("grad_pooled.bin", (4, 8))
cpp_g_conv = load_bin("grad_conv_out.bin", (4, 8, 26, 26))
cpp_g_w = load_bin("grad_weights.bin", (8, 1, 3, 3))
cpp_g_in = load_bin("grad_input.bin", (4, 1, 28, 28))

py_in = cpp_in.clone().detach().requires_grad_(True)
py_w  = cpp_w.clone().detach().requires_grad_(True)
py_fc_w = cpp_fc_w.clone().detach().requires_grad_(True)

py_conv = F.conv2d(py_in, py_w, padding=0)
py_conv.retain_grad()

py_pooled = F.adaptive_avg_pool2d(py_conv, (1, 1)).view(4, 8)
py_pooled.retain_grad() 

# Matrix multiplication without bias
py_logits = torch.matmul(py_pooled, py_fc_w)
py_logits.retain_grad()

py_loss = F.cross_entropy(py_logits, cpp_l)
py_loss.backward()

print("\nEXHAUSTIVE GRADIENT VERIFICATION\n" + "="*50)
compare("Forward Pass (Logits)", cpp_fwd, py_logits)
print("-" * 50)
compare("Grad Logits (Softmax-CE)", cpp_g_logits, py_logits.grad)
compare("Grad FC Weights (Matmul)", cpp_g_fc_w, py_fc_w.grad)
compare("Grad Pooled (Matmul A)", cpp_g_pooled, py_pooled.grad)
compare("Grad Conv Output (Pooling)", cpp_g_conv, py_conv.grad)
compare("Grad Weights (Conv)", cpp_g_w, py_w.grad)
compare("Grad Input (Conv)", cpp_g_in, py_in.grad)