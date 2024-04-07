import torch
import torch.nn.functional as F 
from collections import OrderedDict
from functools import partial
import warnings

DEBUG = False

class MetaMonkey(torch.nn.Module):
    """Trace a networks and then replace its module calls with functional calls.

    This allows for backpropagation w.r.t to weights for "normal" PyTorch networks.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.parameters = OrderedDict(net.named_parameters())
        
    def forward(self, inputs, parameters=None):
        if parameters is None:
            return self.net(inputs)
        
        param_gen = iter(parameters.values())
        method_pile = []
        counter = 0
        
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                ext_weight = next(param_gen)
                if module.bias is not None:
                    ext_bias = next(param_gen)
                else:
                    ext_bias = None
                
                method_pile.append(module.forward)
                module.forward = partial(F.conv2d, weight=ext_weight, bias=ext_bias, stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups)
            
            elif isinstance(module, torch.nn.Linear):
                lin_weights = next(param_gen)
                lin_bias = next(param_gen)
                method_pile.append(module.forward)
                module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)   
            
            elif next(module.parameters(), None) is None:
                pass
            
            else:
                # Warn for other containers
                if DEBUG:
                    warnings.warn(f'Patching for module {module.__class__} is not implemented.')
        
        output = self.net(inputs)
        
        # undo patching
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                module.forward = method_pile.pop(0)
            elif isinstance(module, torch.nn.Linear):
                module.forward = method_pile.pop(0)
        
        return output