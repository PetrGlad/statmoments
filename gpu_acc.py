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
n = 3  # Number of scalar values in a measurement
p = 2  # Max moment degree to calc

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
print("> x=\n", x)
x_powers = cp.zeros((n, p), dtype=dtype)

block_size = min(ck_powers.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
  n_blocks += 1
print(n_blocks, block_size)

ck_powers((n_blocks,), (block_size,), (p, x, x_powers))  # grid (number of blocks), block and arguments
print("> x_powers=\n", x_powers)


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
void pairs_update(const int n, const int p, const int* acc_row_indexes, const T* x_powers, T* acc) {
  // Using `_i` as shorthand for "index", `i` alone means row index.
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  printf("<%d (b=%d, t=%d)> # %d %d \n", tid, blockIdx.x, threadIdx.x, n, p);
  const int cell_mat_size = p * p;
  const int row_i = tid;
  const int next_row_i = acc_row_indexes[row_i + 1];
  int acc_i = acc_row_indexes[row_i];
  int col_i = 0;
  // TODO (scheduling) Updating a single row of the accumulator for now. 
  //      Should be a contiguous range of rows
  const int powers_idx_a = row_i * p;
  int powers_idx_b = 0;
  // For each pair cell of the triangle matrix row.
  printf("|before acc_i %d, next_row_i %d\n", acc_i, next_row_i);
  for (; acc_i < next_row_i; col_i++) {
    printf("|row acc_i %d, col_i %d\n", acc_i, col_i);
    for (int pa = 0; pa < p; pa++) {
      for (int pb = 0; pb < p; pb++) {
        printf("|cell tid %d, acc_i %d, col_i %d, pa %d, pb %d, xp1 %f, xp2 %f\n",
               tid, acc_i, col_i, pa, pb, x_powers[powers_idx_a + pa],  x_powers[powers_idx_b + pb]);
        acc[acc_i] += x_powers[powers_idx_a + pa] * x_powers[powers_idx_b + pb];
        acc_i++;
      }
    }
    powers_idx_b += p;
  }
}
""",
  "pairs_update",
)


def acc_to_2d_numpy(n_rows, cell_size, acc):
  """Convert in-GPU accumulator representation to 2d numpy array.
     For diagnostics and GPU->host export.
     This may produce output that is easier to inspect than one of acc_to_4d_numpy."""
  out = np.zeros((n_rows * cell_size, n_rows * cell_size), dtype=dtype)
  row_len = 1
  idx = 0
  for i in range(n_rows):
    for j in range(row_len):
      for ci in range(cell_size):
        for cj in range(cell_size):
          out[i * cell_size + ci, j * cell_size + cj] = acc[idx]
          idx += 1
    row_len += 1
  return out


def acc_to_4d_numpy(n_rows, cell_size, acc):
  """Convert in-GPU accumulator representation to a logically laid out numpy array.
     For diagnostics and GPU->host export.
     first 2 indexes select variable pair, second 2 indexes select powers of sums."""
  out = np.zeros((n_rows, n_rows, cell_size, cell_size), dtype=dtype)
  row_len = 1
  idx = 0
  for i in range(n_rows):
    for j in range(row_len):
      for ci in range(cell_size):
        for cj in range(cell_size):
          out[i, j, ci, cj] = acc[idx]
          idx += 1
    row_len += 1
  return out


# n_acc_cols = n
# n_acc_rows = math.ceil(n / 2)

starts = row_starts()
cell_size = p
cell_matrix_size = cell_size * cell_size
row_index = np.array([next(starts) * cell_matrix_size
                      for _ in range(n + 1)], dtype=np.uint32)
print("> row_index=\n", row_index)
assert row_index[-1] == math.floor((n + 1) * n / 2) * cell_matrix_size

row_indexes = cp.array(row_index, dtype=cp.uint32)
acc = cp.zeros((row_index[-1],), dtype=dtype)

block_size = min(ck_pairs_update.max_threads_per_block, n)
n_blocks = n // block_size
if n % block_size != 0:
  n_blocks += 1

print(n_blocks, block_size)
ck_pairs_update((n_blocks,), (block_size,), (n, p, row_indexes, x_powers, acc))  # grid (number of blocks), block size and arguments
print(f"n={n}, p={p}")
print("> acc=\n", acc)
print("> acc=\n", acc_to_2d_numpy(n, cell_size, acc))

print("\nDONE")
