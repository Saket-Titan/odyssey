import numpy as np
class Tensor:
    def __init__(self,data, requires_grad:bool = False):
        self.data = np.asarray(data, dtype = np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = ()
        self._backward = lambda : None
        

    def __add__(self, other):
        if not isinstance(other,Tensor):
            other = Tensor(other)

        return Tensor(self.data + other.data, self.requires_grad or other.requires_grad)