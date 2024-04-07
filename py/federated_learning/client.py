# from federated_learning.data.mnist_processing import get_mnist_dataloaders
from federated_learning.utils import model_to_numpy_dict, numpy_dict_to_model, quantized_lenet_forward
from federated_learning.aggregator import FedAvg
from federated_learning.model import LeNet_Small_Quant, mlleaks_mlp
from .attacks import ModelInversion, MemberInference
from blockchain import Transaction
from tqdm import trange
import numpy as np
import torch
import time
import copy
from torch.utils.data import DataLoader, TensorDataset
from typing import List

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
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.quantized = False
        self.malicous = malicious
        if self.malicous:
            self.init_attacker(attack_type)

    def get_model(self, model_params: dict):
        self.model.load_state_dict(model_params)
        
    def init_attacker(self, attack_type=None):
        if attack_type is None:
            # randomly select an attack type
            attack_type = np.random.choice(['model_inversion', 'membership_inference'])
        
        

    def local_train(self, epochs, lr, batch_size, optimizer, criterion):
        """ 
        Method to train the model locally.
        :param model: Model to be trained.
        :param train_loader: DataLoader for the training dataset.
        :param test_loader: DataLoader for the testing dataset.
        :param epochs: Number of epochs for training.
        :param lr: Learning rate for the optimizer.
        :param device: Device to be used for training.
        """
        if self.model is None:
            raise ValueError("Model not found")

        model = self.model
        X_train = torch.from_numpy(self.X_train).float()
        y_train = torch.from_numpy(self.y_train).long()
        device = self.device
        model.to(device)
        model.train()
        
        train_data = TensorDataset(X_train, y_train)

        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        
        # if self.X_test is not None or self.y_test is not None:
        #     X_test = torch.from_numpy(self.X_test).float()
        #     y_test = torch.from_numpy(self.y_test).long()
        #     test_data = TensorDataset(X_test, y_test)
        #     test_loader = DataLoader(
        #     test_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for data, label in train_loader:
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(
                    f"Client {self.index} Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
            
            # if self.X_test is None or self.y_test is None:
            #     continue
            
            # model.eval()
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for data, label in test_loader:
            #         data, label = data.to(device), label.to(device)
            #         output = model(data)
            #         _, predicted = torch.max(output, 1)
            #         total += label.size(0)
            #         correct += (predicted == label).sum().item()
            #     acc = 100*correct/total
            # print(
            #     f"Client {self.index} Epoch {epoch+1}/{epochs} Accuracy: {acc:.6f}%")

        model.to('cpu')
        
    def evaluate(self, model):
        if self.X_test is None or self.y_test is None:
        # test on the training data
            X_test = torch.from_numpy(self.X_train).float()
            y_test = torch.from_numpy(self.y_train).long()
        else:
            X_test = torch.from_numpy(self.X_test).float()
            y_test = torch.from_numpy(self.y_test).long()
        # make inference
        model = model
        model.eval()
        with torch.no_grad():
            output = model(X_test)
            _, predicted = torch.max(output, 1)
            correct = (predicted == y_test).sum().item()
            total = y_test.size(0)
            accuracy = correct / total
        return accuracy
    
    def local_update(self, eval=True):
        """ 
        Method to update the model locally.
        :param model: Model to be updated.
        :param accuracy: Accuracy of the model on the testing dataset.
        :return: Dict containing the updated model parameters and the accuracy.
        """
        update = {}
        update['worker id'] = self.index
        if eval:
            accuracy = self.evaluate(model=self.model)
        else:
            accuracy = 0.0
        update['accuracy'] = accuracy
        numpy_model = model_to_numpy_dict(self.model)
        update['model_params'] = numpy_model
        update['model'] = copy.deepcopy(self.model)
        return update

    
    def send_tx(self, pool: List[Transaction], eval=True):
        local_update = self.local_update(eval=eval)
        tx = Transaction(
            sender_id = self.index,
            task = 'cifar10',
            model = local_update['model'],
            model_params=local_update['model_params'],
            accuracy=local_update['accuracy'],
            timestamp=time.time(),
            verified=False
        )
        pool.append(tx)
    
    def verify_tx(self, unverified_tx: List[Transaction]):
        """ 
        Used in pofl to verify the transactions by running the model on the testing dataset and see if the accuracy matches.
        If the first transaction is verified, which should have the highest accuracy, stop verifying the rest and return the verified transaction.
        unverified_tx: Sorted List of unverified transactions.
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("No public testing dataset found")
        tx_with_highest_acc = None
        for tx in unverified_tx:
            model = tx.model
            accuracy = self.evaluate(model)
            print(f"worker {self.index} test model from {tx.sender_id}, got accuracy {accuracy}")
            if accuracy == tx.accuracy:
                tx_with_highest_acc = tx
                break
        return tx_with_highest_acc
    
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


    def quantized_model_forward(self, x, dump_flag, dump_path='..zkp/pretrained_model/LeNet_CIFAR_pretrained'):
        path = dump_path + '/worker_' + str(self.index)
        quantized_lenet_forward(model=self.model, x=x, dump_flag=dump_flag, dump_path=path) # dump the quantized model
    
class MIAttacker(Worker):
    """ 
    Membership Inference Attacker
    """
    def __init__(self, shadow_epochs, shadow_loss, shadow_optim, attack_epochs, attack_loss, attack_optim, adv_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shadow_epochs = shadow_epochs
        self.shadow_loss = shadow_loss
        self.shadow_optim = shadow_optim
        self.attack_epochs = attack_epochs
        self.attack_loss = attack_loss
        self.attack_optim = attack_optim
        self.adv_type = adv_type
    
    def adv_1(self, target: Worker):
        """ 
        Attack 1: Train a shadow model to predict the membership of the target model.
        """
        
        
        
    
        
        
def reconstruct_training_data(tx: Transaction, num_images: int, ground_truth: torch.Tensor):
    model = tx.model
    model.eval()
    
    setup = ModelInversion.utils.system_startup()
    defs = ModelInversion.training_strategy('conservative')
    dm = torch.as_tensor(ModelInversion.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(ModelInversion.consts.cifar10_std, **setup)[:, None, None]
    
    model = tx.model # the model that we need to reconstruct the training data
    input_parameters = []
    for name, param in model.named_parameters():
        input_parameters.append(param)
    
    config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=2000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')
    
    rec_machine = ModelInversion.FedAvgReconstructor(model, (dm, ds), local_steps=3, local_lr=1e-3, config=config, use_updates=True, num_images=100)
    output, stats = rec_machine.reconstruct(input_data=input_parameters, labels=None, img_shape=(3, 32, 32))
    
    test_mse = (output.detach() - ground_truth).pow(2).mean()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
    test_psnr = ModelInversion.metrics.psnr(output, ground_truth, factor=1/ds)
    
    return output, test_mse, feat_mse, test_psnr
    