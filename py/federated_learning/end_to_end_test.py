import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from operators.intOperators import FullyConnected, ConvolutionOperator, AvgPoolOperator
from operators.utils import extract_uint_weight, dump_txt
from data_utils import load_cifar10
from federated_learning.model import LeNet_Small_Quant

FC1_in_channel_num = 480

(X_train, Y_train), (X_test, Y_test) = load_cifar10(num_users=1, n_class=10, n_samples=1000, rate_unbalance=1)

model = LeNet_Small_Quant()
weight_file = 'federated_learning/model/worker1.pkl'
model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))

Y_test_preds = torch.argmax(model(torch.tensor(X_test).float()), dim=1).numpy()
print("Floating-point PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# Calibrate
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(128))
    X = torch.tensor(X_train[samp]).float()
    Y = torch.tensor(Y_train[samp]).long()
    out = model(X)

print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

# Evaluate Quantized Model. Accuracy varies across trails.
Y_test_preds = torch.argmax(model(torch.tensor(X_test).float()), dim=1).numpy()
print("Quantized PyTorch Model Accuracy:", (Y_test_preds == Y_test).mean())

weights = model.state_dict()

feature_quantize_parameters = model.dump_feat_param()

DUMP_FLAG = True

model_name = 'LeNet_Small'

def forward(x):
    # First quant on input x.
    x_quant_int_repr, x_quant_scale, x_quant_zero_point = model.quant_input(x)
    if DUMP_FLAG == True:
        dump_txt(x_quant_int_repr, x_quant_zero_point, x_quant_scale, 'federated_learning\layerInfo/X')    
    # 1st layer 
    q1, z1, s1 = extract_uint_weight(weights['conv1.weight'])
    q2, z2, s2 = x_quant_int_repr, x_quant_zero_point, x_quant_scale
    z3, s3 = feature_quantize_parameters['conv1_q_zero_point'], feature_quantize_parameters['conv1_q_scale']
    z3 = 128
    conv1 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'federated_learning\layerInfo/'+model_name+'_conv1_weight')
        dump_txt(conv1, z3, s3, 'federated_learning\layerInfo/'+model_name+'_conv1_output')
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
        dump_txt(q1, z1, s1 * s2 / s3, 'federated_learning\layerInfo/'+model_name+'_conv2_weight')
        dump_txt(conv2, z3, s3, 'federated_learning\layerInfo/'+model_name+'_conv2_output')
    act2 = np.maximum(conv2, z3)
    # pool2, s3, z3 = SumPoolOperator(act2, 2), s3/4, z3*4
    pool2 = AvgPoolOperator(act2, 2)
    if DUMP_FLAG == True:
        dump_txt(pool2, z3, s3, 'federated_learning\layerInfo/'+model_name+'_avgpool2_output')
    # 3rd layer
    q1, z1, s1 = extract_uint_weight(weights['conv3.weight'])
    q2, z2, s2 = pool2, z3, s3
    z3, s3 = feature_quantize_parameters['conv3_q_zero_point'], feature_quantize_parameters['conv3_q_scale']
    z3 = 128
    conv3 = ConvolutionOperator(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'federated_learning\layerInfo/'+model_name+'_conv3_weight')
        dump_txt(conv3, z3, s3, 'federated_learning\layerInfo/'+model_name+'_conv3_output')
    act3 = np.maximum(conv3, z3)

    view_output = act3.reshape((-1,FC1_in_channel_num))

    # 4th layer
    q1, z1, s1 = extract_uint_weight(weights['linear1._packed_params._packed_params'][0])
    q2, z2, s2 = view_output, z3, s3
    z3, s3 = feature_quantize_parameters['linear1_q_zero_point'], feature_quantize_parameters['linear1_q_scale']
    z3 = 128
    linear1 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'federated_learning\layerInfo/'+model_name+'_linear1_weight')
        dump_txt(linear1, z3, s3, 'federated_learning\layerInfo/'+model_name+'_linear1_output')
    act4 = np.maximum(linear1, z3)

    # 5th layer
    q1, z1, s1 = extract_uint_weight(weights['linear2._packed_params._packed_params'][0])
    q2, z2, s2 = act4, z3, s3
    z3, s3 = feature_quantize_parameters['linear2_q_zero_point'], feature_quantize_parameters['linear2_q_scale']
    z3 = 128
    linear2 = FullyConnected(q1, z1, s1, q2, z2, s2, z3, s3)
    if DUMP_FLAG == True:
        dump_txt(q1, z1, s1 * s2 / s3, 'federated_learning\layerInfo/'+model_name+'_linear2_weight')
        dump_txt(linear2, z3, s3, 'federated_learning\layerInfo/'+model_name+'_linear2_output')
    return linear2

# eval
# print("X_test.shape: ", X_test.shape, ', Y_test.shape: ', Y_test.shape)
# Y_test_preds_out = forward(X_test[:1000].reshape((-1, 3, 32, 32)))
# Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
# np.savetxt(model_name+"_classification.txt", Y_test_preds.flatten(), fmt='%u', delimiter=',')

# print("Quantized Numpy Model Accuracy: ", (Y_test[:1000] == Y_test_preds).mean())