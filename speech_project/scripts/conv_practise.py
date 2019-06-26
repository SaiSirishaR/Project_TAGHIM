import torch
import torch.nn as nn

m = nn.Conv1d(5, 10, 3, stride=2)
input = torch.randn(7, 5, 6)
output = m(input)
print("ouput is", output)
