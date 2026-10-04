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
        out =  Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data,dtype = np.float32)
                self.grad += out.grad*1.0
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other,dtype = np.float32)
                other.grad += out.grad*1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)
        


    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or self.requires_grad)
        out._prev = (self, other)
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data, dtype=np.float32)
                self.grad += other.data*out.grad


            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data,dtype = np.float32)
                
                other.grad += self.data*out.grad


        out._backward = _backward

        return out

    def __rmul__(self,other):
        return self.__mul__(other)

         
        



        


        