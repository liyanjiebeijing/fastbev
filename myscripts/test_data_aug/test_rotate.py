import torch
import math
import numpy as np



p = torch.tensor([[1.0, 0.0, 0.0]])
theta = torch.tensor(math.pi/2)
rot = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta),  torch.cos(theta), 0],
                    [0, 0, 1]])

print(p @ rot)
print(rot @ p.T)
