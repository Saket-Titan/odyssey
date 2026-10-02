import numpy as np
class Tensor:
    def __init__(self,data, requires_grad:bool = False):
        self.data = np.asarray(data, dtype = np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = None
        self._backward = None
        