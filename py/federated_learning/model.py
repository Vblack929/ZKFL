import numpy as np
import torch
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class FLModel(torch.nn.Module):
    def __init__(self, device):
        super(FLModel, self).__init__()
        self.device = device
        self.optim = None
        self.loss_fn = None

    def set_optimizer(self, optim):
        self.optim = optim

    def get_params(self):
        return [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]

    def set_params(self, params):
        """Set the model parameters to the given values.

        Args:
            params : List of numpy arrays containing the model parameters.
        """
        with torch.no_grad():
            for p, p_new in zip(self.parameters(), params):
                p.copy_(torch.tensor(p_new))

    def forward(self, x):
        pass

    def calc_acc(self, x):
        pass

    def train_step(self, x, y):
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.calc_acc(logits, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), acc

    def eval_step(self, x, y, B):
        """ 
        Args:
            x: input data   {torch.tensor}
            y: target labels {torch.tensor}
            B: batch size   {int}
        """
        n_batches = int(np.ceil(x.shape[0] / B))
        err = 0.0
        acc = 0.0

        with torch.no_grad():
            for b in range(n_batches):
                logits = self.forward(x[b*B:(b+1)*B])
                err += self.loss_fn(logits, y[b*B:(b+1)*B]).item()
                acc += self.calc_acc(logits, y[b*B:(b+1)*B])

        return err / n_batches, acc / n_batches


class LeNet_Small_Quant(FLModel):
    """
    Lenet model with 3 convolutional layers and 2 fully connected layers 
    for CIFAR-10 dataset. The model is quantized using PyTorch's quantization
    """

    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        super(LeNet_Small_Quant, self).__init__(device=device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False)
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=480, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        x = self.dequant(x)
        return x

    def calc_acc(self, logits, y):
        return torch.mean((torch.argmax(logits, dim=1) == y).float()).item()

    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 3, 32, 32))).float()
        x = self.quant(dummy_image)
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        conv2 = self.conv2(pool1)
        act2 = self.act2(conv2)
        pool2 = self.pool2(act2)
        conv3 = self.conv3(pool2)
        act3 = self.act3(conv3)
        view_output = act3.reshape(act3.size(0), -1)
        linear1 = self.linear1(view_output)
        act4 = self.act4(linear1)
        linear2 = self.linear2(act4)
        output = self.dequant(linear2)
        feature_quantize_parameters = {'x_q_scale': x.q_scale(), 'x_q_zero_point': x.q_zero_point(),
                                       "conv1_q_scale": conv1.q_scale(), 'conv1_q_zero_point': conv1.q_zero_point(),
                                       'act1_q_scale': act1.q_scale(), "act1_q_zero_point": act1.q_zero_point(),
                                       'pool1_q_scale': pool1.q_scale(), 'pool1_q_zero_point': pool1.q_zero_point(),
                                       'conv2_q_scale': conv2.q_scale(), "conv2_q_zero_point": conv2.q_zero_point(),
                                       "act2_q_scale": act2.q_scale(), "act2_q_zero_point": act2.q_zero_point(),
                                       'pool2_q_scale': pool2.q_scale(), "pool2_q_zero_point": pool2.q_zero_point(),
                                       "conv3_q_scale": conv3.q_scale(), "conv3_q_zero_point": conv3.q_zero_point(),
                                       "act3_q_scale": act3.q_scale(), "act3_q_zero_point": act3.q_zero_point(),
                                       "linear1_q_scale": linear1.q_scale(), "linear1_q_zero_point": linear1.q_zero_point(),
                                       "act4_q_scale": act4.q_scale(), "act4_q_zero_point": act4.q_zero_point(),
                                       "linear2_q_scale": linear2.q_scale(), "linear2_q_zero_point": linear2.q_zero_point()
                                       }
        return feature_quantize_parameters

    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()


class LeNet_Small(nn.Module):
    def __init__(self):
        super(LeNet_Small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1, bias=False)
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False)
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False)
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=480, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        return x


class mlleaks_mlp(nn.Module):
    def __init__(self, n_in=3, n_out=1, n_hidden=64):
        super(mlleaks_mlp, self).__init__()
        self.hidden = nn.Linear(n_in, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        out = self.out(x)
        return out
