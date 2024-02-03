import torch
import time
import numpy as np


def compare():
    mps_time = 1
    nomps_time = 0
    initial_size = 100000
    size = initial_size
    while nomps_time < mps_time:
        # Use MPS
        start = time.time()
        mps_device = torch.device("mps")
        x = torch.ones(size, device=mps_device, dtype=torch.float32)
        y = x * 2
        end = time.time()
        mps_time = end - start

        # Don't use MPS
        start = time.time()
        xn = np.ones(size, dtype=np.float32)
        yn = xn * 2
        end = time.time()
        nomps_time = end - start
        size += initial_size

    print("break even size:", size)


compare()
