import numpy as np

def get_activation_utils(type):
    match type:
        case 'tanh':
            f_fun = np.tanh
            grad_f_wr_z_fun = lambda z: 1 - np.power(f_fun(z), 2)
            grad_fun_method = 'element-wize'
        case 'sigmoid':
            f_fun = sigmoid
            grad_f_wr_z_fun = sigmoid_prim
            grad_fun_method = 'element-wize'
        case 'softmax':
            f_fun = softmax
            grad_f_wr_z_fun = softmax_diff
            grad_fun_method = 'matmul'      
        case _:
            raise Exception("Undefined Type")
    return f_fun, grad_f_wr_z_fun, grad_fun_method


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_prim(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def softmax_diff(z):
    s = softmax(z)
    return np.diag(s.reshape(-1,)) - np.matmul(s, s.T)


def get_criterion_utils(type):
    match type:
        case 'SE':
            C_fun = lambda y, y_hat: (1/(y.shape[0]))*sum(np.power(y - y_hat, 2))
            grad_C_wr_y_hat_fun = lambda y, y_hat: (y_hat - y).T
        case 'CE':
            C_fun = lambda y, y_hat: -sum(np.multiply(y, np.log(y_hat)))
            grad_C_wr_y_hat_fun = lambda y, y_hat: -np.divide(y, y_hat).T
        case 'CCE':
            raise Exception("Not Implemented")
        case _:
            raise Exception("Undefined Type")
    return C_fun, grad_C_wr_y_hat_fun



def CCE_after_softmax(y, y_hat):
    return -sum(np.multiply(y, np.log(y_hat)))

def CCE_after_softmax_diff(y, y_hat):
    return (y_hat - y).T
