import torch
import numpy as np
import copy
from .operators.intOperators import FullyConnected, ConvolutionOperator, AvgPoolOperator


def from_torch_to_numpy(tensor):
    """
    Convert a PyTorch tensor to a numpy array.
    """
    return tensor.cpu().detach().numpy()


def from_numpy_to_torch(array):
    """
    Convert a numpy array to a PyTorch tensor.
    """
    return torch.from_numpy(array)

def model_to_numpy_dict(model):
    """
    Convert a PyTorch model to a dictionary of numpy arrays.
    """
    numpy_model = {}
    for key, value in model.state_dict().items():
        numpy_model[key] = from_torch_to_numpy(value)
    return numpy_model

def numpy_dict_to_model(numpy_model, model_struct):
    """
    Convert a dictionary of numpy arrays to a PyTorch model.
    """
    model = copy.deepcopy(model_struct)
    for key, value in numpy_model.items():
        model.state_dict()[key].copy_(from_numpy_to_torch(value))
    return model

def extract_uint_weight(weight):
    q = weight.int_repr().numpy().astype(np.int32)
    z = weight.q_per_channel_zero_points().numpy().astype(np.int32)
    s = weight.q_per_channel_scales().numpy()
    assert (z == np.zeros(z.shape)).all(), 'Warning: zero point is not zero'
    z = 128 
    q += 128
    q = q.astype(np.int32)
    return q, z, s

def dump_txt(q, z, s, prefix):
    np.savetxt(prefix + '_q.txt', q.flatten(), fmt='%u', delimiter=',')
    f1 = open(prefix + '_z.txt', 'w+')
    if(str(z)[0] == '['):
        f1.write(str(z)[1:-1])
    else:
        f1.write(str(z))
    f1.close()
    f2 = open(prefix + '_s.txt', 'w+')
    if(str(s)[0] == '['):
        f2.write(str(s)[1:-1])
    else:
        f2.write(str(s))
    f2.close()
    
def quantized_lenet_forward(model, x, y, dump_flag, dump_path):
    model_name = 'LeNet_Small'
    model = model
    weights = model.state_dict()
    feature_quantize_parameters = model.dump_feat_param()
    DUMP_FLAG = dump_flag
    FC1_in_channel_num = 480
    path = dump_path

    x_quant_int_repr, x_quant_scale, x_quant_zero_point = model.quant_input(
        x)
    if DUMP_FLAG == True:
        dump_txt(x_quant_int_repr, x_quant_zero_point,
                    x_quant_scale, path + 'X')
    # 1st layer
    q1, z1, s1 = extract_uint_weight(weights['conv1.weight'])
    q2, z2, s2 = x_quant_int_repr, x_quant_zero_point, x_quant_scale
    z3, s3 = feature_quantize_parameters['conv1_q_zero_point'], feature_quantize_parameters['conv1_q_scale']
    z3 = 128
    conv1 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, path + model_name+'_conv1_weight')
        dump_txt(conv1, z3, s3, path + model_name+'_conv1_output')
    act1 = np.maximum(conv1, z3)
    # pool1, s3, z3 = SumPoolOperator(act1, 2), s3/4, z3*4
    pool1 = AvgPoolOperator(act1, 2)

    # 2nd layer
    q1, z1, s1 = extract_uint_weight(weights['conv2.weight'])
    q2, z2, s2 = pool1, z3, s3
    z3, s3 = feature_quantize_parameters['conv2_q_zero_point'], feature_quantize_parameters['conv2_q_scale']
    z3 = 128
    conv2 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, path + model_name+'_conv2_weight')
        dump_txt(conv2, z3, s3, path + model_name+'_conv2_output')
    act2 = np.maximum(conv2, z3)
    # pool2, s3, z3 = SumPoolOperator(act2, 2), s3/4, z3*4
    pool2 = AvgPoolOperator(act2, 2)
    if DUMP_FLAG == True:
        dump_txt(pool2, z3, s3, path + model_name+'_avgpool2_output')
    # 3rd layer
    q1, z1, s1 = extract_uint_weight(weights['conv3.weight'])
    q2, z2, s2 = pool2, z3, s3
    z3, s3 = feature_quantize_parameters['conv3_q_zero_point'], feature_quantize_parameters['conv3_q_scale']
    z3 = 128
    conv3 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, path + model_name+'_conv3_weight')
        dump_txt(conv3, z3, s3, path + model_name+'_conv3_output')
    act3 = np.maximum(conv3, z3)

    view_output = act3.reshape((-1, FC1_in_channel_num))

    # 4th layer
    q1, z1, s1 = extract_uint_weight(
        weights['linear1._packed_params._packed_params'][0])
    q2, z2, s2 = view_output, z3, s3
    z3, s3 = feature_quantize_parameters['linear1_q_zero_point'], feature_quantize_parameters['linear1_q_scale']
    z3 = 128
    linear1 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, path + model_name+'_linear1_weight')
        dump_txt(linear1, z3, s3, path + model_name+'_linear1_output')
    act4 = np.maximum(linear1, z3)

    # 5th layer
    q1, z1, s1 = extract_uint_weight(
        weights['linear2._packed_params._packed_params'][0])
    q2, z2, s2 = act4, z3, s3
    z3, s3 = feature_quantize_parameters['linear2_q_zero_point'], feature_quantize_parameters['linear2_q_scale']
    z3 = 128
    linear2 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, path + model_name+'_linear2_weight')
        dump_txt(linear2, z3, s3, path + model_name+'_linear2_output')
        
    # eval
    y_test_pred = np.argmax(linear2, axis=1)
    np.savetxt(path + 'classification.txt', y_test_pred.flatten(), fmt='%u', delimiter=',')
    eval_acc = np.mean(y_test_pred == y)

    return eval_acc