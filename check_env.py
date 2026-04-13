# # save as check_env.py
# import os
# import sys
# import subprocess
# import torch

# print("Python:", sys.version)
# print("Torch:", torch.__version__)
# print("Torch CUDA:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     print("Current device:", torch.cuda.current_device())
#     print("Device name:", torch.cuda.get_device_name(0))
#     print("Capability:", torch.cuda.get_device_capability(0))

#     x = torch.randn(2, 3, device="cuda")
#     y = x * 2
#     print("Simple CUDA tensor ok:", y.shape, y.dtype, y.device)

#     try:
#         free_mem, total_mem = torch.cuda.mem_get_info()
#         print("GPU mem free / total (GB):", round(free_mem/1024**3, 2), "/", round(total_mem/1024**3, 2))
#     except Exception as e:
#         print("mem_get_info failed:", repr(e))

# print("\n=== nvidia-smi ===")
# try:
#     out = subprocess.check_output(["nvidia-smi"], text=True)
#     print(out)
# except Exception as e:
#     print("nvidia-smi failed:", repr(e))

import torch
import torch.nn as nn

print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())

x = torch.randn(1, 3, 8, 32, 32, device="cuda", dtype=torch.float16)
conv = nn.Conv3d(3, 16, 3, padding=1).to("cuda", dtype=torch.float16)

with torch.no_grad():
    y = conv(x)

print("OK:", y.shape, y.dtype, y.device)