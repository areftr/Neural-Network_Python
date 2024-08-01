import numpy as np
from ..functional import get_activation_utils, get_criterion_utils


class Layer:
    def __init__(self):
        self.name = None
        self.input = None

    def forward(self, input):
        raise NotImplemented
    
    def backward(self, errr):
        raise NotImplemented


class FCLayer(Layer):
    def __init__(self, ni, no, name=None, initialization='random', activation='sigmoid'):
        self.ni = ni
        self.no = no
        self.name = name

        if initialization not in ['zero', 'random'] and not isinstance(initialization, dict):
            raise Exception("initialization must be 'zero', 'random', or a dictionary")

        if initialization == 'zero':
            # Perform zero initialization
            W = np.zeros([no, ni], dtype=float)
            b = np.zeros([no, 0], dtype=float)
        elif initialization == 'random':
            W = np.random.rand(no, ni)
            b = np.random.rand(no, 0)
        elif isinstance(initialization, dict):
            # Use the provided dictionary for initialization
            if 'W' in initialization and 'b' in initialization:
                W = initialization['W'].reshape(no, ni)
                b = initialization['b'].reshape(no, -1)
            else:
                raise Exception("Dictionary must contain 'W' and 'b' keys")        
        self.W = W
        self.b = b

        f_fun, grad_f_wr_z_fun, grad_fun_method = get_activation_utils(activation)
        self.f_fun = f_fun
        self.grad_f_wr_z_fun = grad_f_wr_z_fun
        self.grad_fun_method = grad_fun_method

        self.grad_J_wr_W = np.zeros_like(W)
        self.grad_J_wr_b = np.zeros_like(b)
        self.z = None
        self.a = None
    
    def forward(self, input):        
        self.input = input
        self.z = np.matmul(self.W, input) + self.b
        return  self.f_fun(self.z)
    
    def backward(self, deltaW):
        match self.grad_fun_method:
            case 'element-wize':
                DELTAW = np.multiply(self.grad_f_wr_z_fun(self.z), deltaW)
            case 'matmul':
                DELTAW = np.matmul(self.grad_f_wr_z_fun(self.z).T, deltaW)
            case _:
                raise Exception("Undefined Method for Actication Gradient Evaluation")
        
        grad_C_wr_W = np.multiply(DELTAW, self.input.T)
        self.grad_J_wr_W += grad_C_wr_W
        self.grad_J_wr_b += DELTAW
        return np.matmul(self.W.T, DELTAW)
    
    def loose_grad(self):
        self.grad_J_wr_W = np.zeros_like(self.W)
        self.grad_J_wr_b = np.zeros_like(self.b)
    
    def update(self, N, epsilon, alpha):
        self.grad_J_wr_W = self.grad_J_wr_W/N + alpha*self.W
        self.grad_J_wr_b = self.grad_J_wr_b/N
        self.W -= epsilon*self.grad_J_wr_W
        self.b -= epsilon*self.grad_J_wr_b


class FFNet:
    def __init__(self):
        pass

    def predict(self, x):
        input_l = x
        for layer in self.layer_array:
            output_l = layer.forward(input_l)
            input_l = output_l
        return output_l

    def get_sumW(self):
        output = 0
        for layer in self.layer_array:
            output += np.sum(layer.W)
        return output

    def get_sumsquareW(self):
        output = 0
        for layer in self.layer_array:
            output += np.sum(np.power(layer.W, 2))
        return output
    
    def update(self, N, epsilon, alpha):
        for layer in self.layer_array:
            layer.update(N, epsilon, alpha)

    def zero_grads(self):
        for layer in self.layer_array:
            layer.loose_grad()


class Criterion():
    def __init__(self, criterion):
        self.criterion = criterion
        C_fun, grad_C_wr_y_hat = get_criterion_utils(criterion)  
        self.evaluate = C_fun
        self.grad_C_wr_y_hat_fun = grad_C_wr_y_hat

class Optimizer():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def cal_grads(self, y, y_hat):
        deltaW_l = self.criterion.grad_C_wr_y_hat_fun(y, y_hat).T
        for layer in reversed(self.model.layer_array):
            deltaW_l = layer.backward(deltaW_l)   