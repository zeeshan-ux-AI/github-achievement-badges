
# Enhancement entry 10
# FP8 Block Quantization Module
import torch
def quantize_fp8(x):
    return x.to(torch.float8_e4m3fn)

