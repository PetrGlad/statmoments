class GpuAccumulator:
  pass


class GpuEngine:
  def __init__(self, trace_len, cl_len, moment=2, normalize=True, acc_min_count=10, **_options):
    assert moment >= 0
    self.moment = moment  # max co-moment degree
    assert trace_len > 0
    self.trace_len = trace_len  # Number of variables in a measurement.
    self.tags_count = cl_len # not used fo far, expected self.tags_count <= len(self.accs)
    self.normalize = normalize
    self.total_count = 0  # Total number of measurements across all accumulators?
    self.min_measurement_count = acc_min_count
    self.accs = {}  # tag -> accumulator

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    raise NotImplementedError("GpuEngine::estimate_mem_size")

  def memory_size(self):
    raise NotImplementedError("GpuEngine::memory_size")

  def update(self, traces, tags):
    assert len(traces) == len(tags)
    raise NotImplementedError("GpuEngine::update")

  def counts(self, i):
    raise NotImplementedError("GpuEngine::counts")

  def moments(self, moments=None, normalize=None):
    """Return a generator yielding a pair of univariate statistical moments for each classifier"""
    # moments = moments if moments is not None else [self.moment]
    # normalize = normalize if normalize is not None else self.normalize
    # return self._moments(moments, normalize)
    raise NotImplementedError("GpuEngine::moments")

  # def _ensure_ret(self, mlen):
  #   if self._retm.shape[1] < mlen:
  #     tr_len = self.trace_len
  #     self._retm = np.empty((2, mlen, tr_len * (tr_len + 1) // 2), dtype=np.float64)
  #   return self._retm

  def comoments(self, moments=None, normalize=None):
    """Return a generator yielding the requested set of bivariate comoments for each classifier"""
    # moments = moments if moments is not None else np.broadcast_to(self.moment, (2, 1))
    # normalize = normalize if normalize is not None else self.normalize
    # if len(moments[0]) != len(moments[1]):
    #   raise ValueError("The input moment lists should have equal lengths.")
    #
    # if self.moment < np.max(moments):
    #   raise ValueError("The input moment should be less or equal than indicated in constructor.")
    #
    # self._ensure_ret(len(moments[0]))
    #
    # return self._comoments(moments, normalize)
    raise NotImplementedError("GpuEngine::comoments")
