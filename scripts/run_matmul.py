import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

with open("compiled/kernel.ptx", "r") as f:
    ptx = f.read()
mod = cuda.module_from_buffer(ptx.encode("utf-8"))
func = mod.get_function("function")

m = 4096
n = 4096
k = 1024
block_size_m = 32
block_size_n = 32

a = np.random.randn(m, k).astype(np.float32)
b = np.random.randn(k, n).astype(np.float32)
c = np.zeros((m, n), dtype=np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

grid_size_m = (m + block_size_m - 1) // block_size_m
grid_size_n = (n + block_size_n - 1) // block_size_n
block = (block_size_m, block_size_n, 1)
grid = (grid_size_m, grid_size_n, 1)
t0 = time.perf_counter()
func(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(n), np.int32(k), block=block, grid=grid)
t1 = time.perf_counter()
latency = t1 - t0
print(f"Latency: {latency*1000:.2f} ms")

cuda.memcpy_dtoh(c, c_gpu)

c_true = a @ b
print(np.allclose(c, c_true, atol=1e-3))

c_error = np.abs(c - c_true)
plt.imshow(c_error, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.savefig("matmul.png")
