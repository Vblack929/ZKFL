import numpy as np

def NaiveConv(data, weight):
    """ 
    Numpy implementation of a convolutional layer.
    :param data: input data, numpy array of shape [N, N]
    :param weight: convolutional kernel, numpy array of shape [kernel_size, kernel_size]
    """
    assert data.shape[0] == data.shape[1]   
    assert weight.shape[0] == weight.shape[1]
    N = data.shape[0]
    kernel_size = weight.shape[0]
    output = np.ones((N - kernel_size + 1, N - kernel_size + 1))
    for x in range(N - kernel_size + 1):
        for y in range(N - kernel_size + 1):
            output[x, y] = np.sum(data[x:x + kernel_size, y:y + kernel_size] * weight)
    return output

def GeneralConv(data, weight, bias=None):
    """ 
    Numpy implementation of a convolutional layer.
    :param data: input data, numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the height and width of the input data.
    :param weight: convolutional kernel, numpy array of shape [K, C, kernel_size, kernel_size]. K is the number of kernels, C is the number of channels, kernel_size is the size of the kernel.
    :return: output data, numpy array of shape [N, K, H-kernel_size+1, W-kernel_size+1].
    """
    assert weight.shape[2] == weight.shape[3]
    N, C, H, W = data.shape
    K, C, kernel_size, kernel_size = weight.shape
    output = np.ones((N, K, H - kernel_size + 1, W - kernel_size + 1)).astype(np.single)
    for n in range(N):
        for x in range(H - kernel_size + 1):
            for y in range(W - kernel_size + 1):
                for k in range(K):
                    output[n, k, x, y] = np.sum(data[n, :, x:x + kernel_size, y:y + kernel_size] * weight[k])
    if bias is not None:
        for k in range(K):
            output[:, k, :, :] += bias[k]
    
    return output

def conv(data, weight, bias=None):
    """ 
    Wrapper function for the convolutional layer.
    :param data: input data, numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the height and width of the input data.
    :param weight: convolutional kernel, numpy array of shape [K, C, kernel_size, kernel_size]. K is the number of kernels, C is the number of channels, kernel_size is the size of the kernel.
    :return: output data, numpy array of shape [N, K, H-kernel_size+1, W-kernel_size+1].
    """
    return GeneralConv(data, weight, bias)

def avg_pool(data, kernel_size):
    """ 
    Numpy implementation of an average pooling layer.
    :param data: numpy array of shape [N, C, H, W]. N is the batch size, C is the number of channels, H and W are the height and width of the input data.
    :param kernel_size: size of the pooling kernel.
    :return: output data, numpy array of shape [N, C, H//kernel_size, W//kernel_size].
    """
    N, C, H, W = data.shape
    output = np.ones((N, C, H//kernel_size, W//kernel_size)).astype(np.single)  
    for n in range(N):
        for c in range(C):
            for x in range(H//kernel_size):
                for y in range(W//kernel_size):
                    output[n, c, x, y] = np.mean(data[n, c, x*kernel_size:(x+1)*kernel_size, y*kernel_size:(y+1)*kernel_size])
    return output

if __name__ == "__main__":
    data = np.ones((1, 1, 28, 28))
    weight = np.ones((32, 1, 3, 3))
    output = conv(data, weight)
    
    print("output: ", output)