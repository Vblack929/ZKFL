# from federated_learning.data.mnist_processing import get_mnist_dataloaders
# Standard library imports
import copy
import time
from typing import List

# Related third party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from tqdm import trange

# Local application/library specific imports
from federated_learning.utils import quantized_lenet_forward
from federated_learning.model import FLModel
from blockchain import Transaction
# from .attacks import ModelInversion, MemberInference

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps"


class Worker:
    def __init__(self, index, X_train, y_train, X_test, y_test, model,  malicious=False, attack_type=None):
        """ 
        Constructor for the `Worker` class.
        :param dataset: Dataset to be used for training.
        :param model: Model to be trained.
        """
        self.index = index
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset = TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).long())
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.quantized = False
        self.dump_path = ''
        self.malicous = malicious
        if self.malicous:
            self.init_attacker(attack_type)

    def set_params(self, new_params):
        """ 
        Args:
            new_params: New model parameters as a list of Numpy arrays.
        """
        self.model.set_params(new_params)
        
    def get_params(self):
        return self.model.get_params()
    
    def set_optimizer(self, optimizer):
        self.model.set_optimizer(optimizer)
        
        
    # def init_attacker(self, attack_type=None):
    #     if attack_type is None:
    #         # randomly select an attack type
    #         attack_type = np.random.choice(['model_inversion', 'membership_inference'])
        
        

    def train_step(self, model, K, B):
        """ 
        Args:
            model: FLModel object.
            K: Number of local epochs.
            B: Batch size.
        """
        train_loader = DataLoader(self.dataset, batch_size=B, shuffle=True)
        model.to(self.device)
        model.train()
        for k in range(1, K+1):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss, _ = model.train_step(x, y)
            if k % 10 == 0:
                print(f"Worker {self.index} local epoch {k}: loss {loss}")
                
    def train_step_dp(self, model, K, B, norm, eps, delta):
        train_loader = DataLoader(self.dataset, batch_size=B, shuffle=True)
        model.to(self.device)   
        model.train()
        # create privacy engine
        privacy_engine = PrivacyEngine()
        
        model, optim, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=model.optim,
            data_loader=train_loader,
            epochs=K,
            target_epsilon=eps,
            target_delta=delta,
            max_grad_norm=norm,
        )
        
        for k in range(1, K+1):
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optim.step()
                losses.append(loss.item())
            
            if k % 10 == 0:
                print(f"Worker {self.index} local epoch {k}: loss {np.mean(losses)}")
                
        
    def evaluate(self, model, x, y, B):
        """
        Args:
            model: FLModel object.
            x: input data.
            y: target labels.
            B: batch size.
        """
        x = torch.tensor(x).float().to(self.device)
        y = torch.tensor(y).long().to(self.device)
        model.to(self.device)
        model.eval()
        err, acc = model.eval_step(x, y, B)
        return err, acc
        
    
    def local_update(self, acc):
        """ 
        Returns the local update of the worker. {accuracy, model_params, model}
        """
        update = {}
        update['worker id'] = self.index
        update['accuracy'] = acc
        update['params'] = self.get_params()
        return update

    
    def send_tx(self, local_update: dict, pool: List[Transaction]):
        tx = Transaction(
            sender_id = local_update['worker id'],
            task = 'cifar10',
            params=local_update['params'],
            accuracy=local_update['accuracy'],
            timestamp=time.time(),
            verified=False
        )
        pool.append(tx)
    
    def quantize_model(self):
        if device == "mps":
            self.model.qconfig = torch.quantization.get_default_qconfig(
                'qnnpack')
            torch.backends.quantized.engine = 'qnnpack'
            print(self.model.qconfig)
            torch.quantization.prepare(self.model, inplace=True)
            # Calibrate the model
            for i in (t := trange(10)):
                samp = np.random.randint(
                    0, len(self.dataset['train'].dataset), size=(64))
                X = torch.tensor(self.X_train[samp]).float()
                Y = torch.tensor(self.y_train[samp]).long()
                out = self.model(X)
            print("Calibration done")
            torch.quantization.convert(self.model, inplace=True)
            print("Quantization done")

        elif device == "cuda":
            self.model.qconfig = torch.quantization.get_default_qconfig(
                'fbgemm')
            print(self.model.qconfig)
            torch.quantization.prepare(self.model, inplace=True)
            # Calibrate the model
            for i in (t := trange(10)):
                samp = np.random.randint(
                    0, len(self.dataset['train'].dataset), size=(64))
                X = torch.tensor(self.X_train[samp]).float()
                Y = torch.tensor(self.y_train[samp]).long()
                out = self.model(X)
            print("Calibration done")
            torch.quantization.convert(self.model, inplace=True)
            print("Quantization done")

        self.quantized = True


    def quantized_model_forward(self, x, dump_flag, dump_path='pretrained_model/LeNet_CIFAR_pretrained'):
        quantized_lenet_forward(model=self.model, x=x, dump_flag=dump_flag, dump_path=dump_path) # dump the quantized model
    
# class MIAttacker(Worker):
#     """ 
#     Membership Inference Attacker
#     """
#     def __init__(self, shadow_epochs, shadow_loss, shadow_optim, attack_epochs, attack_loss, attack_optim, adv_type, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.shadow_epochs = shadow_epochs
#         self.shadow_loss = shadow_loss
#         self.shadow_optim = shadow_optim
#         self.attack_epochs = attack_epochs
#         self.attack_loss = attack_loss
#         self.attack_optim = attack_optim
#         self.adv_type = adv_type
    
#     def adv_1(self, target: Worker):
#         """ 
#         Attack 1: Train a shadow model to predict the membership of the target model.
#         """
        
        
        
    
        
        
# def reconstruct_training_data(tx: Transaction, num_images: int, ground_truth: torch.Tensor):
#     model = tx.model
#     model.eval()
    
#     setup = ModelInversion.utils.system_startup()
#     defs = ModelInversion.training_strategy('conservative')
#     dm = torch.as_tensor(ModelInversion.consts.cifar10_mean, **setup)[:, None, None]
#     ds = torch.as_tensor(ModelInversion.consts.cifar10_std, **setup)[:, None, None]
    
#     model = tx.model # the model that we need to reconstruct the training data
#     input_parameters = []
#     for name, param in model.named_parameters():
#         input_parameters.append(param)
    
#     config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.1,
#               optim='adam',
#               restarts=1,
#               max_iterations=2000,
#               total_variation=1e-6,
#               init='randn',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')
    
#     rec_machine = ModelInversion.FedAvgReconstructor(model, (dm, ds), local_steps=3, local_lr=1e-3, config=config, use_updates=True, num_images=100)
#     output, stats = rec_machine.reconstruct(input_data=input_parameters, labels=None, img_shape=(3, 32, 32))
    
#     test_mse = (output.detach() - ground_truth).pow(2).mean()
#     feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
#     test_psnr = ModelInversion.metrics.psnr(output, ground_truth, factor=1/ds)
    
#     return output, test_mse, feat_mse, test_psnr
    