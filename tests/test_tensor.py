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

def test_tensor_scalar_Addition():
    a = Tensor([1.0,2.0],requires_grad=True)

    c1 = a + 5.0
    c2 = 5.0 + a
    assert np.allclose(c1.data, [6.0, 7.0])
    assert np.allclose(c2.data, [6.0, 7.0])


def test_tensor_mul_forward_and_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    c.grad = np.ones_like(c.data)
    c._backward()
    assert np.allclose(c.data,[8.0, 15.0])
    assert np.allclose(b.grad,[2.0,3.0])
    assert np.allclose(a.grad,[4.0,5.0])


    d = a * 5.0
    a.grad = None
    d.grad = np.ones_like(d.data)
    d._backward()
    assert np.allclose(d.data,[10.0, 15.0])
    assert np.allclose(a.grad,[5.0,5.0])

    e = 5.0 * a
    a.grad = None
    e.grad = np.ones_like(e.data)
    e._backward()
    assert np.allclose(e.data,[10.0, 15.0])
    assert np.allclose(a.grad,[5.0,5.0])
    
    f = a * a
    a.grad = None
    f.grad = np.ones_like(f.data)
    f._backward()
    assert np.allclose(f.data,[4.0, 9.0])
    assert np.allclose(a.grad,[4.0,6.0])


def test_tensor_backward_chain():
    x = Tensor([2.0,3.0], requires_grad=True)
    w1 = Tensor([3.0, 4.0],requires_grad=True)
    w2 = Tensor([1.0, 5.0],requires_grad=True)

    a = x*w1
    b = a + w2

    L = b*2.0

    L.backward()

    assert np.allclose(L.grad,[1.0 , 1.0])
    assert np.allclose(b.grad,[2.0 , 2.0])
    assert np.allclose(w2.grad,[2.0 , 2.0])
    assert np.allclose(a.grad,[2.0 , 2.0])
    assert np.allclose(w1.grad,[4.0 , 6.0])
    assert np.allclose(x.grad,[6.0 , 8.0])


def test_tensor_backward_diamond():
    x = Tensor([3.0 -1.0], requires_grad=True)
    a = x*3.0
    b = x*5.0
    L = a + b
    L.backward()

    assert np.allclose(x.grad,[8.0 , 8.0])
