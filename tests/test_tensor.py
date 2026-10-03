import numpy as np
from odyssey.tensor import Tensor


def test_tensor_initialization():
    t = Tensor([1.0,2.0,55])

    assert isinstance(t.data, np.ndarray)
    assert t.data.dtype == np.float32
    assert t.grad is None
    assert t.requires_grad is False
    assert t._prev == ()


def test_tensor_add_forward_and_backward():

    a = Tensor([2.0,3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=False)

    c = a + b

    assert isinstance(c,Tensor)
    assert np.allclose(c.data, [6.0,8.0])
    assert c.requires_grad is True
    assert c._prev == (a,b)

    c.grad = np.ones_like(c.data)
    c._backward()

    assert np.allclose(a.grad, [1.0,1.0])
    assert b.grad is None

    d = a + a
    d.grad = np.ones_like(d.data)
    a.grad = None
    d._backward()
    assert np.allclose (a.grad, [2.0,2.0])

