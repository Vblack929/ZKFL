import blockchain
import federated_learning
from federated_learning.client import Worker
from federated_learning.model import LeNet_Small_Quant
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import zkfl
import warnings

warnings.filterwarnings("ignore")

EPS = 0.05


class Network():
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        self.consensus = ''
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
                                                                                   )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        if model.lower() == 'lenet':
            self.global_model = LeNet_Small_Quant()
        self.path = "../chains" + self.consensus + "/"

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

        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        path = os.path.join(path, t)

        os.makedirs(path)

        # initialize the blockchain
        self.blockchain = blockchain.Blockchain(consensus=self.consensus, max_nodes=self.num_clients, model=self.global_model,
                                                task='cifar10', save_path=path)
            
class POFLNetWork(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(num_clients, global_rounds, local_rounds, frac_malicous, dataset, model)
        self.consesus = 'pofl'
        self.init_network(clear_path=False)
        
    def run(self):
        # init workers
        self.workers = []
        global_accuracy = []
        X_test, y_test = self.X_test[:1000], self.y_test[:1000] # public test set
        for i in range(self.num_clients):
            if i <= self.num_malicous:
                worker = Worker(index=i+1,
                                X_train=self.X_train[i],
                                y_train=self.y_train[i],
                                X_test=None,
                                y_test=None,
                                model=LeNet_Small_Quant(),
                                malicious=True)
            else:
                worker = Worker(index=i+1,
                                X_train=self.X_train[i],
                                y_train=self.y_train[i],
                                X_test=None,
                                y_test=None,
                                model=LeNet_Small_Quant(),
                                malicious=False)
            self.workers.append(worker)
            
        
        for i in range(1, self.global_rounds+1):
            print(f"Global round {i}")
            # workers load global params from the last block
            global_params = self.blockchain.last_block.global_params
            for w in self.workers:
                w.model = LeNet_Small_Quant()
                w.set_params(global_params)
            # local training
            self.local_train(B=128)
            print("Local training done")
            # evaluate and send tx
            for worker in self.workers:
                _, acc = worker.evaluate(model=worker.model, x=X_test, y=y_test, B=64)
                update = worker.local_update(acc=acc)
                worker.send_tx(update, self.blockchain.transaction_pool) 
            print("Transactions sent")
            # eval update
            print("Start eval update")
            self.blockchain.sort_transactions()
            for tx in self.blockchain.transaction_pool:
                vote = 0
                params = tx.params
                for worker in self.workers:
                    worker.set_params(params)
                    _, acc = worker.evaluate(model=worker.model, x=X_test, y=y_test, B=64)
                    if acc == tx.accuracy:
                        print(f"transaction from worker {tx.sender_id} verified by worker {worker.index}")
                        vote += 1
                    else:
                        print(f"transaction from worker {tx.sender_id} rejected by worker {worker.index}")
                if vote == len(self.workers):
                    tx.verified = True
                    print(f"transaction from worker {tx.sender_id} verified by all workers")
                    # worker who sent this tx becomes the leader
                    leader_id = tx.sender_id
                    print(f"worker {leader_id} is the leader")
                    break
            
            leader = [worker for worker in self.workers if worker.index == leader_id][0]
            # leader perform aggregation
            new_block = blockchain.Block(index=len(self.blockchain),
                                        transactions=self.blockchain.transaction_pool,
                                        timestamp=time.time(),
                                        previous_hash=self.blockchain.last_block.hash,
                                        global_params=None,
                                        )
            new_block.miner_id = leader_id
            # aggregate
            print("Start aggregation")
            agg = federated_learning.FedAvg(global_model=self.global_model, beta=0.9, lr=0.1)
            new_global_params = agg.aggregate(local_params=[tx.params for tx in new_block.transactions]).get_params()
            # eval global model
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(model=leader.model, x=X_test, y=y_test, B=64)
            new_block.global_params = new_global_params
            new_block.global_accuracy = gloabl_acc
            global_accuracy.append(gloabl_acc)
            # append block to blockchain
            self.blockchain.add_block(new_block)
            if not self.blockchain.valid_chain:
                print("Chain invalid")
                break
            self.blockchain.store_block(new_block)
            self.blockchain.empty_transaction_pool()
        
        # save the global accuracy as txt
        np.savetxt(self.blockchain.save_path + '/global_accuracy.txt', np.array(global_accuracy))

        plt.plot(global_accuracy)
        plt.xlabel("Global rounds")
        plt.ylabel("Global accuracy")
        plt.show()
            
            
    def local_train(self, B):
        for worker in self.workers:
            worker.model.set_optimizer(torch.optim.Adam(worker.model.parameters(), lr=0.001))
            worker.train_step_dp(
                model=worker.model,
                K=self.local_rounds,
                B=B,
                norm=1.2,
                eps=50.0,
                delta=1e-5,
            )
        
        
            
class ZKFLChain(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(num_clients, global_rounds, local_rounds, frac_malicous, dataset, model)
        self.consesus = 'zkfl'
        self.log = {}
        
    def run(self):
        self.workers = []
        X_test, y_test = self.X_test[:1000], self.y_test[:1000] # public test set
        for i in range(self.num_clients):
            if i <= self.num_malicous:
                worker = Worker(index=i+1,
                                X_train=self.X_train[i],
                                y_train=self.y_train[i],
                                X_test=None,
                                y_test=None,
                                model=LeNet_Small_Quant(),
                                malicious=True)
            else:
                worker = Worker(index=i+1,
                                X_train=self.X_train[i],
                                y_train=self.y_train[i],
                                X_test=None,
                                y_test=None,
                                model=LeNet_Small_Quant(),
                                malicious=False)
            self.workers.append(worker)
            
        for i in range(1, self.global_rounds+1):
            print(f"Global round {i}")
            # workers load global params from the last block
            global_params = self.blockchain.last_block.global_params
            for w in self.workers:
                w.set_params(global_params)
            # local training
            self.local_train(B=64)
            print("Local training done")
            # quantize model
            for worker in self.workers:
                worker.quantize_model()
                dump_path = f'../pretrained_models/worker_{worker.index}/'
                # clear the path if not empty
                if os.path.exists(dump_path):
                    for file in os.listdir(dump_path):
                        file_path = os.path.join(dump_path, file)
                        os.remove(file_path)
                else:
                    os.makedirs(dump_path)
                worker.quantized_model_forward(x=X_test, dump_flag=True, dump_path=dump_path)
                worker.dump_path = dump_path

            # eval and generate proof
            for worker in self.workers:
                acc = zkfl.generate_proof(worker.dump_path)
                update = worker.local_update(acc=acc)
                worker.send_tx(update, self.blockchain.transaction_pool)
            
            self.blockchain.sort_transactions()
            leader_id = self.blockchain.transaction_pool[0].sender_id
            leader = [worker for worker in self.workers if worker.index == leader_id][0]
            # leader perform aggregation
            new_block = blockchain.Block(index=len(self.blockchain),
                                        transactions=self.blockchain.transaction_pool,
                                        timestamp=time.time(),
                                        previous_hash=self.blockchain.last_block.hash,
                                        global_params=None,
                                        )
            new_block.miner_id = leader_id
            agg = federated_learning.FedAvg(global_model=self.global_model, beta=0.9, lr=0.1)
            new_global_params = agg.aggregate(local_params=[tx.params for tx in new_block.transactions]).get_params()
            # eval global model
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(model=leader.model, x=X_test, y=y_test, B=64)
            new_block.global_params = new_global_params
            new_block.global_accuracy = gloabl_acc
            # append block to blockchain
            self.blockchain.add_block(new_block)
            if not self.blockchain.valid_chain:
                print("Chain invalid")
                break
            self.blockchain.store_block(new_block)
            self.blockchain.empty_transaction_pool()
                
            
    def local_train(self, B):
        for worker in self.workers:
            worker.set_optimizer(torch.optim.Adam(worker.model.parameters(), lr=0.001))
            worker.train_step(
                model=worker.model,
                K=self.local_rounds,
                B=B
            )

def vanillia_fl(num_clients, global_rounds, local_rounds):
    """ 
    A simple federated learning network with no malicious clients.
    """
    (X_train, y_train), (X_test, y_test) = federated_learning.load_cifar10(num_users=num_clients,
                                                                           n_class=10,
                                                                           n_samples=1000,
                                                                           rate_unbalance=1.0,
                                                                           )
    workers = []
    for i in range(num_clients):
        worker = Worker(index=i+1,
                        X_train=X_train[i],
                        y_train=y_train[i],
                        X_test=None,
                        y_test=None,
                        model=LeNet_Small_Quant(),
        )
        workers.append(worker)
        
    global_model = LeNet_Small_Quant() 
    global_accuracy = []
    for i in range(1, global_rounds+1):
        global_params = global_model.get_params()
        local_params = []
        for w in workers:
            w.set_params(global_params)
            w.set_optimizer(torch.optim.Adam(w.model.parameters(), lr=0.001))
            w.train_step(
                model=w.model,
                K=local_rounds,
                B=128
            )
            local_params.append(w.get_params())
        agg = federated_learning.FedAvg(global_model=global_model, beta=0.9, lr=0.1)
        new_global_model = agg.aggregate(local_params=local_params)
        _, acc = new_global_model.eval_step(x=X_test, y=y_test, B=64)
        print(f"Global round {i}: accuracy {acc}")
        global_accuracy.append(acc)
        global_model.set_params(new_global_model.get_params())
    
    # plot the global accuracy
    plt.plot(global_accuracy)
    plt.xlabel("Global rounds")
    plt.ylabel("Global accuracy")
    plt.show()
            
    
            
        
        
    
if __name__ == '__main__':
    network = POFLNetWork(
        num_clients=5,
        global_rounds=50,
        local_rounds=20,
        frac_malicous=0.0,
        dataset='cifar10',
        model='lenet'
    )
    network.run()
    
    