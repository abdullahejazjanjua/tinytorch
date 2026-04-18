import struct
import torch
import torch.nn.functional as F
import numpy as np

N, in_C, in_H, in_W = 1, 3, 5, 5
out_C, _, k_H, k_W = 2, 3, 3, 3
padding = 1

x = torch.randn(N, in_C, in_H, in_W, requires_grad=True, dtype=torch.float32)
w = torch.randn(out_C, in_C, k_H, k_W, requires_grad=True, dtype=torch.float32)

out = F.conv2d(x, w, padding=padding)
dout = torch.randn_like(out, dtype=torch.float32)
out.backward(dout)


def export_bin(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(data.tobytes())

export_bin(x, '../data/input.bin')
export_bin(w, '../data/filters.bin')
export_bin(out, '../data/output_ref.bin') 
export_bin(dout, '../data/dout.bin')
export_bin(w.grad, '../data/grad_w_ref.bin')

metadata = struct.pack('7i', N, in_C, in_H, in_W, out_C, k_H, padding)
with open('../data/meta.bin', 'wb') as f:
    f.write(metadata)