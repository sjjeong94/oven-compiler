import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

with open("compiled/kernel.ptx", "r") as f:
    ptx = f.read()
mod = cuda.module_from_buffer(ptx.encode("utf-8"))
func = mod.get_function("function")

m = 8192
n = 8192
k = 4096
block_size = 32

a = np.random.randn(m, k).astype(np.float32)
b = np.random.randn(k, n).astype(np.float32)
c = np.zeros((m, n), dtype=np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

grid_size_row = (m + block_size - 1) // block_size
grid_size_col = (n + block_size - 1) // block_size
block = (block_size, block_size, 1)
grid = (grid_size_col, grid_size_row, 1)
shared_bytes = block_size * block_size * 4 * 2
t0 = time.perf_counter()
func(a_gpu, b_gpu, c_gpu, np.int32(m), np.int32(n), np.int32(k), block=block, grid=grid, shared=shared_bytes)
t1 = time.perf_counter()
latency = t1 - t0
print(f"Latency: {latency*1000:.2f} ms")

cuda.memcpy_dtoh(c, c_gpu)

c_true = a @ b
if np.allclose(c, c_true, atol=1e-3):
    print("Results are close enough")
else:
    print("Results are not close enough")

c_error = np.abs(c - c_true)
plt.imshow(c_error, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.savefig("matmul.png")
