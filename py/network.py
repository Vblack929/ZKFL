import blockchain
import federated_learning
from federated_learning.client import Worker, reconstruct_training_data
from federated_learning.model import LeNet_Small, LeNet_Small_Quant
from federated_learning.attacks import ModelInversion
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import zkfl


class Network():
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        self.consensus = 'pol'
        self.num_clients = num_clients
        self.global_rounds = global_rounds
        self.local_rounds = local_rounds
        self.frac_malicous = frac_malicous
        self.num_malicous = int(self.num_clients * self.frac_malicous)
        self.dataset = dataset
        if self.dataset.lower() == 'cifar10':
            (X_train, y_train), (X_test, y_test) = federated_learning.load_cifar10(num_users=self.num_clients,
                                                                                   n_class=10,
                                                                                   n_samples=1000,
                                                                                   rate_unbalance=1.0,
                                                                                   even_split=False)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        if model.lower() == 'lenet':
            self.model = LeNet_Small_Quant()
        self.path = "chains/" + self.consensus + "/"

    def init_network(self, clear_path=False):
        path = self.path
        if clear_path:
            for subfolder in os.listdir(path):
                subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    os.remove(file_path)
                os.rmdir(subfolder_path)

        t = time.strftime("%m-%d-%H", time.localtime())
        path = os.path.join(path, t)

        os.makedirs(path)

        # initialize the blockchain
        self.blockchain = blockchain.Blockchain(consensus=self.consensus, max_nodes=self.num_clients, model_struct=self.model,
                                                task='cifar10', save_path=path)
        
    def add_clients(self):
        self.workers = []
        for i in range(self.num_clients):
            if i < self.num_malicous:
                worker = Worker(index=i+1, X_train=self.X_train[i], y_train=self.y_train[i], X_test=None, y_test=None, model=deepcopy(self.model),malicious=True)
            else:
                worker = Worker(index=i+1, X_train=self.X_train[i], y_train=self.y_train[i], X_test=None, y_test=None, model=deepcopy(self.model))
            self.workers.append(worker)
            self.blockchain.register_client(worker)
    
    def local_train_update(self, num_epochs=5):
        # get the latest model from the last block
        latest_block = self.blockchain.last_block
        latest_model = federated_learning.numpy_dict_to_model(numpy_model=latest_block.global_params, model_struct=self.model)
        for worker in self.blockchain.peers:
            worker.model.load_state_dict(latest_model.state_dict()) # load the latest model
            worker.local_train(epochs=num_epochs, lr=0.01, batch_size=32)
            
class POFLNetWork(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(num_clients, global_rounds, local_rounds, frac_malicous, dataset, model)
        self.consesus = 'pofl'
        
    def eval_update(self):
        for worker in self.blockchain.peers:
            worker.X_test, worker.y_test = self.X_test[:100], self.y_test[:100]
            worker.send_tx(self.blockchain.transaction_pool, eval=True)
        
        print("local updates sent")
        print("Sart verification")
        self.blockchain.sort_transactions()
        for tx in self.blockchain.transaction_pool:
            vote = 0
            for worker in self.blockchain.peers:
                acc = worker.evaluate(tx.model)
                if acc == tx.accuracy:
                    print(f"transaction from worker {tx.sender_id} verified by worker {worker.index}")
                    vote += 1
                else:
                    print(f"transaction from worker {tx.sender_id} not verified by worker {worker.index}")
            if vote == len(self.blockchain.peers):
                tx.verified = True
                print(f"transaction from worker {tx.sender_id} verified by all workers")
                # worker who sent this tx becomes the leader
                leader_id = tx.sender_id
                print(f"worker {leader_id} is the leader")
                break
            
        leader = [worker for worker in self.blockchain.peers if worker.index == leader_id][0]
        # leader perform aggregation
        new_block = blockchain.Block(index=len(self.blockchain)+1,
                                     transactions=self.blockchain.transaction_pool,
                                     timestamp=time.time(),
                                     previous_hash=self.blockchain.last_block.hash,
                                     )
        new_block.miner_id = leader_id
        self.blockchain.aggregate_models(new_block)
        global_model = federated_learning.numpy_dict_to_model(numpy_model=new_block.global_params, model_struct=self.model)
        global_acc = leader.evaluate(model=global_model)
        new_block.global_accuracy = global_acc
        print(f"Global model accuracy: {global_acc}")
        self.blockchain.add_block(new_block)
        self.blockchain.store_block(new_block)
        self.blockchain.empty_transaction_pool()
        
            
class ZKFLChain(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(num_clients, global_rounds, local_rounds, frac_malicous, dataset, model)
        self.consesus = 'zkfl'
        self.log = {}
        
    def eval_update(self):
        """ 
        Local clients quantize their models and use int models for evaluation
        """
        for worker in self.workers:
            worker.quantize_model()
            worker.X_test, worker.y_test = self.X_test[:100], self.y_test[:100]
            dump_path = "zkp/pretrained_model" + '_' + str(worker.index)
            print(f"Worker {worker.index} evaluating model")
            eval_acc = worker.quantized_model_forward(x=worker.X_test, y=worker.y_test, dump_flag=True, dump_path=dump_path)
            print(f"Worker {worker.index} model accuracy: {eval_acc}")
    
    def generate_proof(self):
        pass
    
if __name__ == '__main__':
    path = 'pretrained_model/LeNet_CIFAR_pretrained'
    zkfl.read_model(path)
    