from copy import deepcopy
import warnings
import time
import os

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from opacus import PrivacyEngine

import zkfl
import blockchain
import federated_learning
from federated_learning.client import Worker
from federated_learning.model import LeNet_Small_Quant

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
        self.path = "chains" + self.consensus + "/"

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
        super().__init__(num_clients, global_rounds,
                         local_rounds, frac_malicous, dataset, model)
        self.consesus = 'pofl'
        self.init_network(clear_path=False)

    def run(self):
        # init workers
        self.workers = []
        global_accuracy = []
        # public test set
        X_test, y_test = self.X_test[:1000], self.y_test[:1000]
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
                _, acc = worker.evaluate(
                    model=worker.model, x=X_test, y=y_test, B=128)
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
                    _, acc = worker.evaluate(
                        model=worker.model, x=X_test, y=y_test, B=128)
                    if acc == tx.accuracy:
                        print(
                            f"transaction from worker {tx.sender_id} verified by worker {worker.index}")
                        vote += 1
                    else:
                        print(
                            f"transaction from worker {tx.sender_id} rejected by worker {worker.index}")
                if vote == len(self.workers):
                    tx.verified = True
                    print(
                        f"transaction from worker {tx.sender_id} verified by all workers")
                    # worker who sent this tx becomes the leader
                    leader_id = tx.sender_id
                    print(f"worker {leader_id} is the leader")
                    break

            leader = [
                worker for worker in self.workers if worker.index == leader_id][0]
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
            agg = federated_learning.FedAvg(
                global_model=self.global_model)
            new_global_params = agg.aggregate(
                local_params=[tx.params for tx in new_block.transactions])
            # eval global model
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(
                model=leader.model, x=X_test, y=y_test, B=64)
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
        np.savetxt(self.blockchain.save_path +
                   '/global_accuracy.txt', np.array(global_accuracy))

        # plt.plot(global_accuracy)
        # plt.xlabel("Global rounds")
        # plt.ylabel("Global accuracy")
        # plt.show()
        return global_accuracy

    def local_train(self, B):
        for worker in self.workers:
            worker.model.set_optimizer(torch.optim.Adam(
                worker.model.parameters(), lr=0.0001))
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
        super().__init__(num_clients, global_rounds,
                         local_rounds, frac_malicous, dataset, model)
        self.consesus = 'zkfl'
        self.init_network(clear_path=False)

    def run(self):
        self.workers = []
        acc_dict = {}
        global_accuracy = []
        # public test set
        X_test, y_test = self.X_test[:100], self.y_test[:100]
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
            acc_dict[i] = {}
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
                dump_path = f'pretrained_models/worker_{worker.index}/'
                # # clear the path if not empty
                if os.path.exists(dump_path):
                    for file in os.listdir(dump_path):
                        file_path = os.path.join(dump_path, file)
                        os.remove(file_path)
                    # pass
                else:
                    os.makedirs(dump_path)
                    acc = worker.quantized_model_forward(
                        x=X_test, y=y_test, dump_flag=True, dump_path=dump_path)
                    acc_dict[i][worker.index] = acc
                worker.dump_path = dump_path

            # eval and generate proof
            for worker in self.workers:
                acc = float(zkfl.generate_proof(worker.dump_path))
                print(f"worker {worker.index} acc ", acc)
                # if acc == acc_dict[i][worker.index]:
                #     print(f"Worker {worker.index} proof verified")
                # else:
                #     print(f"Worker {worker.index} proof rejected")
                update = worker.local_update(acc=acc)
                worker.send_tx(update, self.blockchain.transaction_pool)

            self.blockchain.sort_transactions()
            leader_id = self.blockchain.transaction_pool[0].sender_id
            leader = [
                worker for worker in self.workers if worker.index == leader_id][0]
            # leader perform aggregation
            new_block = blockchain.Block(index=len(self.blockchain),
                                         transactions=self.blockchain.transaction_pool,
                                         timestamp=time.time(),
                                         previous_hash=self.blockchain.last_block.hash,
                                         global_params=None,
                                         )
            new_block.miner_id = leader_id
            agg = federated_learning.FedAvg(
                global_model=self.global_model)
            new_global_params = agg.aggregate(
                local_params=[tx.params for tx in new_block.transactions])
            # eval global model
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(
                model=leader.model, x=X_test, y=y_test, B=128)
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
        
        return global_accuracy

    def local_train(self, B):
        for worker in self.workers:
            worker.set_optimizer(torch.optim.Adam(
                worker.model.parameters(), lr=0.0001))
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
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).long()
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
            w.set_optimizer(torch.optim.Adam(w.model.parameters(), lr=0.0001))
            w.train_step(
                model=w.model,
                K=local_rounds,
                B=64
            )
            local_params.append(w.get_params())
        agg = federated_learning.FedAvg(
            global_model=global_model)
        new_global_params = agg.aggregate(local_params=local_params)
        global_model.set_params(new_global_params)
        _, acc = global_model.eval_step(x=X_test, y=y_test, B=64)
        print(f"Global round {i}: accuracy {acc}")
        global_accuracy.append(acc)

    # plot the global accuracy
    # plt.plot(global_accuracy)
    # plt.xlabel("Global rounds")
    # plt.ylabel("Global accuracy")
    # plt.show()
    return global_accuracy


def centralized_training(rounds):
    data_dir = 'data/CIFAR10_data/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False)

    model = LeNet_Small_Quant()
    model.set_optimizer(torch.optim.Adam(model.parameters(), lr=0.0001))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model.to(device)
    print("training on ", device)
    train_acc = []
    train_loss = []
    test_loss = []
    test_acc = []
    for k in range(1, rounds+1):
        model.train()
        loss_round = 0.0
        acc_round = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, acc = model.train_step(x, y)
            loss_round += loss
            acc_round += acc
        loss_round /= len(train_loader)
        acc_round /= len(train_loader)
        train_acc.append(acc_round)
        train_loss.append(loss_round)
        if k % 10 == 0:
            print(f"train epoch {k}: loss {loss}")

        # eval
        model.eval()
        loss = 0.0
        acc = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss += model.loss_fn(output, y)
                acc += model.calc_acc(output, y)
        loss /= len(test_loader)
        acc /= len(test_loader)
        test_loss.append(loss.item())
        test_acc.append(acc)
        if k % 10 == 0:
            print(f"test epoch {k}: loss {loss}, acc {acc}")
    # plot the training loss and accuracy in the same figure
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss', color='tab:red')
    # ax1.plot(train_loss, color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Accuracy', color='tab:blue')
    # ax2.plot(train_acc, color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')

    # fig.tight_layout()
    # plt.show()

    # # plot the test loss and accuracy in the same figure
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss', color='tab:red')
    # ax1.plot(test_loss, color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Accuracy', color='tab:blue')
    # ax2.plot(test_acc, color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')

    # fig.tight_layout()
    # plt.show()
    return test_acc


def centralized_dp(rounds: int):
    data_dir = 'data/CIFAR10_data/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False)

    model = LeNet_Small_Quant()
    model.set_optimizer(torch.optim.Adam(model.parameters(), lr=0.0001))
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model.to(device)
    privacy_engine = PrivacyEngine()

    model, optim, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=model.optim,
        data_loader=train_loader,
        epochs=rounds,
        target_epsilon=50.0,
        target_delta=1e-5,
        max_grad_norm=1.2,
    )

    test_loss = []
    test_acc = []
    for k in range(1, rounds+1):
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"epoch {k}: loss {np.mean(losses)}")

        # eval
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = 0.0
        acc = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss += loss_fn(output, y)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = y.detach().cpu().numpy()
                acc += np.sum(preds == labels)

        loss /= len(test_loader)
        acc /= len(test_loader)
        test_loss.append(loss.item())
        test_acc.append(acc)

        print(f"test epoch {k}: loss {loss}, acc {acc}")

    # plot the test accuracy
    plt.plot(test_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def test(rounds: int):
    net1 = POFLNetWork(num_clients=5,
                          global_rounds=rounds,
                          local_rounds=20,
                          frac_malicous=0.0,
                          dataset='cifar10',
                          model='lenet')
    pofl_acc = net1.run()
    net2 = ZKFLChain(num_clients=5,
                        global_rounds=rounds,
                        local_rounds=20, 
                        frac_malicous=0.0, 
                        dataset="cifar10", 
                        model="lenet")
    zkfl_acc = net2.run()
    print(zkfl_acc)
    fl_acc = vanillia_fl(num_clients=5, global_rounds=rounds, local_rounds=20)
    cl_acc = centralized_training(rounds)
    
    # plot all accuracy on one figure
    plt.plot(pofl_acc, label='POFL')
    plt.plot(zkfl_acc, label='ZKFL')
    plt.plot(fl_acc, label='FL')
    plt.plot(cl_acc, label='CL')
    plt.xlabel("Global rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    test(1)
