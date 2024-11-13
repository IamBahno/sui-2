import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {self.value.shape}, got {deltas.shape}'

            # raise NotImplementedError('Backpropagation with deltas not implemented yet')
            self.grad += deltas
            
            # If a back_op exists, propagate the gradient back
            if self.back_op:
                self.back_op.backward(self.grad)
        

        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            # raise NotImplementedError('Backpropagation without deltas not implemented yet')
            # set the initial gradient to be one (because the derivation of L with respect to L is 1)
            self.grad = np.ones((1,1))
            self.back_op.backward(self.grad)


class SuiSumOp:
    def __init__(self,a):
        self.a = a
    def backward(self,grad):
        # broadcast the gradient, into shape of what i am backpropagating to next 
        grad_broadcasted = np.broadcast_to(grad, self.a.value.shape)
        self.a.backward(grad_broadcasted)

def sui_sum(tensor):
    sum_tensor = Tensor(tensor.value.sum(),back_op=SuiSumOp(tensor))
    return sum_tensor


class AddOp:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def backward(self,grad):
        # the derivative of (a+b) with respect to a is 1, and vice versa so
        self.a.backward(grad)
        self.b.backward(grad)

def add(a, b):
    c = a.value + b.value
    tensor = Tensor(c,back_op=AddOp(a,b))
    return tensor

class SubOp:    
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def backward(self,grad):
        # the derivative of (a-b) with respect to a is 1, and vice versa so
        # the derivative of (a-b) with respect to b is -1, and vice versa so
        self.a.backward(grad)
        self.b.backward(-grad)

def subtract(a, b):
    c = a.value - b.value
    tensor = Tensor(c,back_op=SubOp(a,b))
    return tensor

class MultiplyOp:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def backward(self,grad):
        # the derivative of (a*b) with respect to a is b, and vice versa so
        grad_a = grad * self.b.value
        grad_b = grad * self.a.value

        self.a.backward(grad_a)
        self.b.backward(grad_b)

def multiply(a, b):
    c = a.value * b.value
    tensor = Tensor(c,back_op=MultiplyOp(a,b))
    return tensor

class ReluOp:
    def __init__(self,a):
        self.a = a

    def backward(self,grad):
        grad_a = grad * np.where(self.a.value > 0, 1, 0)
        self.a.backward(grad_a)

def relu(tensor):
    out = np.maximum(tensor.value,0)
    return Tensor(out,back_op=ReluOp(tensor))


class DotProductOp:
    def __init__(self,a,b):
        self.a = a
        self.b = b

    def backward(self, grad):
        grad_a = np.dot(grad, self.b.value.T)
        grad_b = np.dot(self.a.value.T, grad)

        self.a.backward(grad_a)
        self.b.backward(grad_b)

def dot_product(a, b):
    c = np.dot(a.value,b.value)
    tensor = Tensor(c,back_op=DotProductOp(a,b))
    return tensor