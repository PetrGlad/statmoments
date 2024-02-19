import unittest
from statmoments._native_shim import dsyrk
import numpy as np
import math


def np_syrk(A, C, alpha, beta, trans, upper, overwrite):
  """syrk reference implementation for comparison."""
  if upper:
    triangle_set = np.triu_indices_from(C, 0)
    triangle_keep = np.tril_indices_from(C, -1)
  else:
    triangle_keep = np.triu_indices_from(C, 1)
    triangle_set = np.tril_indices_from(C, 0)
  if trans:
    A = A.T
  c = alpha * np.matmul(A, A.T) + beta * C
  if overwrite:
    C[triangle_set] = c[triangle_set]
  c[triangle_keep] = C[triangle_keep]
  return c


# Test BLAS implementation bindings
class Test_bindings(unittest.TestCase):

  def test_dsyrk_unit(self):
    A = np.array([[1]], dtype=float)
    C = np.array([[0]], dtype=float)
    dsyrk(A, C, uplo=b'U', trans=b'N', alpha=1.0, beta=1.0)
    c_expected = np.array([[1]], dtype=float)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_dsyrk_coeff(self):
    A = np.array([[3]], dtype=float)
    C = np.array([[1]], dtype=float)
    dsyrk(A, C, uplo=b'U', trans=b'T', alpha=2.0, beta=5.0)
    c_expected = np.array([[23]], dtype=float)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_dsyrk_shape(self):  # SHOULD PASS # 2023-02-21
    A = np.array([[3, 4, 5]], dtype=float)
    C = np.array([[13]], dtype=float)
    dsyrk(A, C, uplo=b'L', trans=b'N', alpha=2.0, beta=5.0)
    c_expected = np.array([[165]], dtype=float)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_dsyrk_shape_2(self):
    A = np.array([[3, 4, 5]], dtype=float)
    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float, order='F')
    dsyrk(A, C, uplo=b'U', trans=b'T', alpha=2.0, beta=5.0)
    c_expected = np.array([[23, 34, 45], [4, 57, 70], [7, 8, 95]], dtype=float)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_dsyrk_shape_2_lower(self):
    A = np.array([[3, 4, 5]], dtype=float)
    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float, order='F')
    print("\nTEST A, C: ", A, C)
    print("TEST A.shape, C.shape: ", A.shape, C.shape)
    dsyrk(A, C, uplo=b'L', trans=b'T', alpha=2.0, beta=5.0)
    c_expected = np.array([[23, 2, 3], [44, 57, 6], [65, 80, 95]], dtype=float)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_dsyrk_shape_2_lower_c_order(self):
    A = np.array([[3, 4, 5]], dtype=np.float64)
    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64, order='F')
    print("\nTEST A, C: ", A, C)
    print("TEST A.shape, C.shape: ", A.shape, C.shape)
    dsyrk(A, C, uplo=b'L', trans=b'T', alpha=2.0, beta=5.0)
    c_expected = np.array([[23, 2, 3], [44, 57, 6], [65, 80, 95]])
    print("\nTEST C out: ", C)
    self.assertTrue(np.array_equal(C, c_expected))

  def test_syrk_reference_comparison(self):
    rng = np.random.default_rng()
    alpha = 3
    beta = 7
    for t in range(10):
      k = 1 + math.floor(rng.random() * 5)
      n = 1 + math.floor(rng.random() * 5)
      upper = rng.choice([b'U', b'L'])
      trans = rng.choice([b'T', b'N'])
      if trans == b'T':
        A = rng.choice([1, 2, 3, 4, 5], (k, n))
      else:
        A = rng.choice([1, 2, 3, 4, 5], (n, k))
      A = A.astype(dtype=np.float64)
      C = rng.choice([1, 2, 3, 4, 5], (n, n)).astype(dtype=np.float64, order='F')
      print("\nC orig=\n", C)
      c_expected = np_syrk(A, C, alpha, beta, upper=upper == b'U', trans=trans == b'T', overwrite=False)
      dsyrk(A, C, uplo=upper, trans=trans, alpha=alpha, beta=beta)
      print("C out=\n", C)
      print("C expected=\n", c_expected)
      self.assertTrue(np.array_equal(C, c_expected))
