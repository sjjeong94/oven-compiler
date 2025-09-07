import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt

with open("compiled/kernel.ptx", "r") as f:
    ptx = f.read()
mod = cuda.module_from_buffer(ptx.encode("utf-8"))
func = mod.get_function("function")

n = 4096 * 2
x = np.random.randn(n).astype(np.float32)
x = np.linspace(-5, 5, n).astype(np.float32)
y = np.zeros_like(x)

x_gpu = cuda.mem_alloc(x.nbytes)
y_gpu = cuda.mem_alloc(y.nbytes)
cuda.memcpy_htod(x_gpu, x)

num_threads = 128
block_size = 128
grid_size = (n + block_size - 1) // block_size
func(x_gpu, y_gpu, block=(num_threads, 1, 1), grid=(grid_size, 1, 1))

cuda.memcpy_dtoh(y, y_gpu)

plt.plot(x, y)
plt.grid()
plt.savefig("kernel.png")