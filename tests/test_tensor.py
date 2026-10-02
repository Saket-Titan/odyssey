import numpy as np
from odyssey.tensor import Tensor


def test_tensor_initialization():
    t = Tensor([1.0,2.0,55])

    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == np.float32
    assert t.grad is None
    assert t.requires_grad is False
    assert t._prev == ()