import itertools as iter
import math

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg.blas as sblas
import sympy as sy
from sympy.plotting import plot

print(cp.cuda.Device(), cp.cuda.device.get_compute_capability(), cp.cuda.get_current_stream())

# Flots to use. Update CUDA kernels when changing this.
dtype = cp.float64
n = 5  # Number of scalar values in a measurement
p = 4  # Max moment degree to calc

ck_powers = cp.RawKernel(
  r"""
typedef double T;

extern "C" __global__
void powers(const int p, const T* x, T* y) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const T z = x[tid];
  T c = z;
  int n = p;
  int i = tid * p;
  while (true) {
      // printf("<%d (b=%d, t=%d)> [%d] at %d = %f # %d\n", tid, blockIdx.x, threadIdx.x, i, n, c, p);
      y[i++] = c;
      if (n <= 1) break;
      n--;
      c *= z;
  }
}
""",
  "powers",
)

print(ck_powers.attributes)

# ---------- x_powers

x = cp.arange(n, dtype=dtype) + 1
print(x)
x_powers = cp.zeros((n, p), dtype=dtype)

bs = min(ck_powers.max_threads_per_block, n)
b = n // bs
if n % bs != 0:
  b += 1
print(b, bs)

ck_powers((b,), (bs,), (p, x, x_powers))  # grid (number of blocks), block and arguments
print(x_powers)


def row_starts():
  p = 0
  i = 0
  while True:
    i += p
    yield i
    p += 1


ck_pairs_update = cp.RawKernel(
  r"""
typedef double T;

extern "C" __global__
void update_pair(const int p, // Currently used also also as the cell size. 
               const int powers_idx_a, const int powers_idx_b, const int acc_idx, 
               const T* x_powers, T* acc) {
               
  // const int tid = blockDim.x * blockIdx.x + threadIdx.x; // DEBUG
  // printf("SUB <%d (b=%d, t=%d)> # %d \n", tid, blockIdx.x, threadIdx.x, p);
  
  for (int pa = 0; pa < p; pa++) {
    for (int pb = 0; pb < p; pb++) {
      // TODO Could probably cast T[p][p] cell = (T[p][p])(&acc[acc_idx]) for convenience.
      acc[acc_idx + pa * p + pb] += x_powers[powers_idx_a + pa] * x_powers[powers_idx_b + pb];
     }
  }
}

extern "C" __global__
void pairs_update(const int n, const int p, const int* row_indexes, const T* x_powers, T* acc) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  printf("<%d (b=%d, t=%d)> # %d %d \n", tid, blockIdx.x, threadIdx.x, n, p);
  
  const int row_i = tid;
  const int next_row_i = row_indexes[row_i + 1];
  int acc_i = row_indexes[row_i];
  int col_i = 0;
  for (; acc_i < next_row_i; acc_i++) { 
    update_pair(p, row_i, col_i, acc_i, x_powers, acc);
  }
}
""",
  "pairs_update",
)


def acc_to_numpy(nrows, acc):
  out = np.zeros((nrows, nrows), dtype=dtype)
  row_len = 1
  idx = 0
  for i in range(nrows):
    for j in range(row_len):
      out[i, j] = acc[idx]
      idx += 1
    row_len += 1
  return out


# n_acc_cols = n
# n_acc_rows = math.ceil(n / 2)

starts = row_starts()
row_index = np.array([next(starts) for _ in range(n + 1)], dtype=np.uint32)
print(row_index)
assert row_index[-1] == math.floor((n + 1) * n / 2)
cell_size = p

row_indexes = cp.array(row_index, dtype=cp.uint32)
acc = cp.zeros((row_index[-1] * cell_size,), dtype=dtype)

bs = min(ck_pairs_update.max_threads_per_block, n)
b = n // bs
if n % bs != 0:
  b += 1
print(b, bs)
ck_pairs_update((b,), (bs,), (n, p, row_indexes, x_powers, acc))  # grid (number of blocks), block size and arguments
print(acc)
print(acc_to_numpy(n, acc))

print("\nDONE")
