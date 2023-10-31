import torch

# Is MPS even available? macOS 12.3+
print("MPS Available:", torch.backends.mps.is_available())

# Was the current version of PyTorch built with MPS activated?
print("PyTorch buil with MPS activated:", torch.backends.mps.is_built())
