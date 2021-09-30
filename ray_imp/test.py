import torch
import numpy as np

if __name__ == "__main__":
    action = 1.2*torch.ones(10)
    actions = [action]*3
    out = torch.as_tensor(np.array(action), dtype=torch.float32)
    a = torch.sub