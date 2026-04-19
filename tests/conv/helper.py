import torch
import numpy as np

def export_bin(tensor, filename):
    data = tensor.detach().cpu().numpy().astype(np.float32)    
    with open(filename, 'wb') as f:
        f.write(data.tobytes())