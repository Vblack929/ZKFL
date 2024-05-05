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
from federated_learning.model import LeNet_Small_Quant, LeNet_MNIST, ShallowNet_Quant

warnings.filterwarnings("ignore")


class Network(blockchain.Blockchain):
    def __init__(self,
                 consensus: str,
                 num_clients: int,
                 global_rounds: int,
                 local_rounds: int,
                 frac_malicous: float,
                 dataset: str,
                 model: str):
        path = "chains"
        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        path = os.path.join(path, t)
        os.makedirs(path)

        if model.lower() == 'lenet':
            self.model = LeNet_Small_Quant()
        elif model.lower() == 'shallownet':
            self.model = LeNet_MNIST()
        super().__init__(consensus=consensus, max_nodes=100,
                         model=self.model, task=dataset, path=path)
        self.num_clients = num_clients
        self.global_rounds = global_rounds
        self.local_rounds = local_rounds
        self.frac_malicous = frac_malicous
        self.num_malicous = int(self.num_clients * self.frac_malicous)
        self.dataset = dataset
        n_class = 10
        n_samples = int(50000 / (self.num_clients * n_class))
        if self.dataset.lower() == 'cifar10':
            (X_train, y_train), (X_test, y_test) = federated_learning.load_cifar10(num_users=self.num_clients,
                                                                                   n_class=n_class,
                                                                                   n_samples=n_samples,
                                                                                   rate_unbalance=1.0,
                                                                                   )
        elif self.dataset.lower() == 'mnist':
            (X_train, y_train), (X_test, y_test) = federated_learning.load_mnist(num_users=self.num_clients)
            # for i in range(len(X_train)):
            #     X_train[i] = X_train[i].reshape(-1, 28 * 28)
            # X_test = X_test.reshape(-1, 28 * 28)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        for i in range(1, self.max_nodes+1):
            if i <= self.num_malicous:
                node = Worker(index=i, X_train=None, y_train=None, X_test=None,
                              y_test=None, model=LeNet_MNIST(), malicious=True, task=self.dataset)
            else:
                node = Worker(index=i, X_train=None, y_train=None, X_test=None,
                              y_test=None, model=LeNet_MNIST(), malicious=False, task=self.dataset)
            self.peers.add(node)

    def init_network(self):
        """ 
        Choose clients for the current round of training
        """
        # randomly select clients from peers, if leader exists from last round, then select the leader
        if self.last_block.miner_id > 0:
            leader = [peer for peer in self.peers if peer.index ==
                      self.last_block.miner_id][0]
            self.workers = [leader]
            # randomly select other clients
            for i in range(self.num_clients-1):
                client = np.random.choice(list(self.peers - set(self.workers)))
                self.workers.append(client)
        else:
            self.workers = np.random.choice(
                list(self.peers), self.num_clients, replace=False)
        # assign data to clients
        global_params = self.last_block.global_params
        for i, worker in enumerate(self.workers):
            worker.X_train = self.X_train[i]
            worker.y_train = self.y_train[i]
            worker.set_params(global_params)


class POFLNetWork(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(consensus='pofl',
                         num_clients=num_clients,
                         global_rounds=global_rounds,
                         local_rounds=local_rounds,
                         frac_malicous=frac_malicous,
                         dataset=dataset,
                         model=model)

    def run(self):
        # init workers
        global_accuracy = []
        # public test set
        X_test, y_test = self.X_test[:1000], self.y_test[:1000]

        for i in range(1, self.global_rounds+1):
            self.init_network()
            print(f"Global round {i}")
            # workers load global params from the last block
            global_params = self.last_block.global_params
            for w in self.workers:
                w.model = LeNet_MNIST()
                w.set_params(global_params)
            # local training
            self.local_train(B=64)
            print("Local training done")
            # evaluate and send tx
            for worker in self.workers:
                _, acc = worker.evaluate(
                    model=worker.model, x=X_test, y=y_test, B=128)
                update = worker.local_update(acc=acc)
                worker.send_tx(update, self.transaction_pool)
            print("Transactions sent")
            # eval update
            print("Start eval update")
            self.sort_transactions()
            for tx in self.transaction_pool:
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
            new_block = blockchain.Block(index=len(self),
                                         transactions=self.transaction_pool,
                                         timestamp=time.time(),
                                         previous_hash=self.last_block.hash,
                                         global_params=None,
                                         )
            new_block.miner_id = leader_id
            # aggregate
            print("Start aggregation")
            agg = federated_learning.FedAvg(
                global_model=self.model)
            new_global_params = agg.aggregate(
                local_params=[tx.params for tx in new_block.transactions])
            # eval global model
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(
                model=leader.model, x=self.X_test, y=self.y_test, B=64)
            new_block.global_params = new_global_params
            new_block.global_accuracy = gloabl_acc
            print(f"global round {i} accuracy: {gloabl_acc}")
            global_accuracy.append(gloabl_acc)
            # append block to blockchain
            self.add_block(new_block)
            if not self.valid_chain:
                print("Chain invalid")
                break
            self.store_block(new_block)
            self.empty_transaction_pool()

        # save the global accuracy as txt
        np.savetxt(self.path +
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
                eps=500.0,
                delta=1e-5,
                noise=1.5
            )


class ZKFLChain(Network):
    def __init__(self, num_clients: int, global_rounds: int, local_rounds: int, frac_malicous: float,
                 dataset: str, model: str):
        super().__init__(consensus='zkfl',
                         num_clients=num_clients,
                         global_rounds=global_rounds,
                         local_rounds=local_rounds,
                         frac_malicous=frac_malicous,
                         dataset=dataset,
                         model=model)

    def run(self):
        log = {}  # used to store workers that failed to generate proof
        acc_dict = {}
        global_accuracy = []
        # public test set
        X_test, y_test = self.X_test[:100], self.y_test[:100]

        for i in range(1, self.global_rounds+1):
            self.init_network()
            acc_dict[i] = {}
            print(f"Global round {i}")
            # workers load global params from the last block
            # local training
            self.local_train(B=128)
            print("Local training done")
            # quantize model
            # for worker in self.workers:
            #     worker.quantize_model()
            #     dump_path = f'pretrained_models/worker_{worker.index}/'
            #     # # clear the path if not empty
            #     if os.path.exists(dump_path):
            #         for file in os.listdir(dump_path):
            #             file_path = os.path.join(dump_path, file)
            #             os.remove(file_path)
            #         # pass
            #     else:
            #         os.makedirs(dump_path)
            #     acc = worker.quantized_model_forward(
            #         x=X_test, y=y_test, dump_flag=True, dump_path=dump_path)
            #     # acc_dict[i][worker.index] = acc
            #     worker.dump_path = dump_path

            # eval and generate proof
            malicious = []
            for worker in self.workers:
                # acc = float(zkfl.generate_proof(worker.dump_path))
                _, acc = worker.evaluate(model=worker.model, x=X_test, y=y_test, B=128)
                claimed_acc = worker.local_update(acc=acc)["accuracy"]
                # print(f"worker {worker.index} acc ", acc)
                # print(f"worker {worker.index} claimed acc ", claimed_acc)
                if acc == claimed_acc:
                    print(f"Worker {worker.index} proof verified")
                else:
                    print(f"Worker {worker.index} proof rejected")
                    malicious.append(worker.index)
                update = worker.local_update(acc=claimed_acc)
                worker.send_tx(update, self.transaction_pool)

            self.sort_transactions()
            # leader would be the first honest worker in the transaction pool
            leader_id = next(
                (tx.sender_id for tx in self.transaction_pool if tx.sender_id not in malicious), None)
            print("leader id", leader_id)
            leader = [
                worker for worker in self.workers if worker.index == leader_id][0]
            # generate proof to test the time
            leader.quantize_model() 
            dump_path = f'pretrained_models/worker_{leader.index}/'
            if os.path.exists(dump_path):
                    for file in os.listdir(dump_path):
                        file_path = os.path.join(dump_path, file)
                        os.remove(file_path)
                    # pass
            else:
                os.makedirs(dump_path)
            start = time.time()
            acc = leader.quantized_model_forward(
                x=X_test, y=y_test, dump_flag=True, dump_path=dump_path)
            end = time.time()
            eval_time = end - start
            print(f"Leader {leader.index} evaluation time: {eval_time}")
            start = time.time()
            generate_acc = float(zkfl.generate_proof(dump_path))
            end = time.time()
            proof_time = end - start
            print(f"Leader {leader.index} proof generation time: {proof_time}")
            # leader perform aggregation
            new_block = blockchain.Block(index=len(self),
                                         transactions=self.transaction_pool,
                                         timestamp=time.time(),
                                         previous_hash=self.last_block.hash,
                                         global_params=None,
                                         )
            new_block.miner_id = leader_id
            if not leader.malicous:  # leader is honest and will perform fedavg aggregation
                agg = federated_learning.FedAvg(
                    global_model=self.model)
            else:  # leader is malicious and will perform guassian attack
                print("Leader is malicious")
                agg = federated_learning.GuassianAttack(
                    global_model=self.model
                )

            new_global_params = agg.aggregate(
                local_params=[tx.params for tx in new_block.transactions])
            leader.model.set_params(new_global_params)
            _, gloabl_acc = leader.evaluate(
                model=leader.model, x=self.X_test, y=self.y_test, B=128)
            print(f"global round {i} accuracy: {gloabl_acc}")
            new_block.global_params = new_global_params
            new_block.global_accuracy = gloabl_acc

            global_accuracy.append(gloabl_acc)
            # append block to blockchain
            self.add_block(new_block)
            if not self.valid_chain:
                print("Chain invalid")
                break
            self.store_block(new_block)
            self.empty_transaction_pool()

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

def fl_dp(num_clients, global_rounds, local_rounds, noise):
    (X_train, y_train), (X_test, y_test) = federated_learning.load_mnist(num_users=20)
    for i in range(len(X_train)):
            X_train[i] = X_train[i].reshape(-1, 28 * 28)
            X_test = X_test.reshape(-1, 28 * 28)
    X_test = torch.tensor(X_test.reshape(-1, 28 * 28)).float()
    y_test = torch.tensor(y_test).long()
    workers = []
    for i in range(num_clients):
        worker = Worker(index=i+1,
                        X_train=X_train[i],
                        y_train=y_train[i],
                        X_test=None,
                        y_test=None,
                        model=ShallowNet_Quant(),
                        )
        workers.append(worker)

    global_model = ShallowNet_Quant()
    global_accuracy = []
    for i in range(1, global_rounds+1):
        torch.cuda.empty_cache()
        global_params = global_model.get_params()
        local_params = []
        for w in workers:
            w.model = ShallowNet_Quant()
            w.set_params(global_params)
            w.set_optimizer(torch.optim.Adam(w.model.parameters(), lr=0.0001))
            w.train_step_dp(
                model=w.model,
                K=local_rounds,
                B=32,
                norm=1.2,
                eps=500.0,
                delta=1e-5,
                noise=noise
            )
            local_params.append(w.get_params())
        agg = federated_learning.FedAvg(
            global_model=global_model)
        new_global_params = agg.aggregate(local_params=local_params)
        global_model.set_params(new_global_params)
        _, acc = global_model.eval_step(x=X_test, y=y_test, B=128)
        print(f"Global round {i}: accuracy {acc}")
        global_accuracy.append(acc)
    return global_accuracy

def vanillia_fl(num_clients, global_rounds, local_rounds):
    """ 
    A simple federated learning network with no malicious clients.
    """
    # (X_train, y_train), (X_test, y_test) = federated_learning.load_cifar10(num_users=num_clients,
    #                                                                        n_class=10,
    #                                                                        n_samples=250,
    #                                                                        rate_unbalance=1.0,
    #                                                                        )
    (X_train, y_train), (X_test, y_test) = federated_learning.load_mnist(num_users=20)
    for i in range(len(X_train)):
            X_train[i] = X_train[i].reshape(-1, 28 * 28)
            X_test = X_test.reshape(-1, 28 * 28)
    X_test = torch.tensor(X_test.reshape(-1, 28 * 28)).float()
    y_test = torch.tensor(y_test).long()
    workers = []
    for i in range(num_clients):
        worker = Worker(index=i+1,
                        X_train=X_train[i],
                        y_train=y_train[i],
                        X_test=None,
                        y_test=None,
                        model=ShallowNet_Quant(),
                        )
        workers.append(worker)

    global_model = ShallowNet_Quant()
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
                B=128
            )
            local_params.append(w.get_params())
        agg = federated_learning.FedAvg(
            global_model=global_model)
        new_global_params = agg.aggregate(local_params=local_params)
        global_model.set_params(new_global_params)
        _, acc = global_model.eval_step(x=X_test, y=y_test, B=128)
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
    # acc = centralized_training(200)
    # # acc = vanillia_fl(num_clients=20, global_rounds=30, local_rounds=5)
    # net = ZKFLChain(num_clients=20,
    #                 global_rounds=100,
    #                 local_rounds=5,
    #                 frac_malicous=0.0,
    #                 dataset='cifar10',
    #                 model='lenet')
    # net = POFLNetWork(num_clients=20,
    #                   global_rounds=200,
    #                   local_rounds=5,
    #                   frac_malicous=0.0,
    #                   dataset='mnist',
    #                   model='shallownet')
    # acc = vanillia_fl(num_clients=20, global_rounds=200, local_rounds=5)
    acc = fl_dp(num_clients=20, global_rounds=200, local_rounds=5, noise=1.5)
    # acc = net.run()
    plt.plot(acc)
    plt.xlabel("Global rounds")
    plt.ylabel("Global accuracy")
    plt.show()
    np.savetxt("dp_mnist1.5.txt", np.array(acc))
    # np.savetxt("fl_mnist.txt", np.array(acc))
    # np.savetxt('cl_200.txt', np.array(acc))
    # np.savetxt('fl_200.txt', np.array(acc))
    # net.run()
    
    # (X_train, Y_train), (X_test, Y_test) = federated_learning.load_mnist(num_users=1)

    # worker = Worker(0, X_train[0], Y_train[0], X_test, Y_test, model=ShallowNet_Quant(), task="mnist")
    # worker.set_optimizer(optimizer=torch.optim.Adam(worker.model.parameters(), lr=0.001))
    # worker.train_step(model=worker.model, K=5, B=64)
    # worker.quantize_model()
    # dump_path = "pretrained_model\shallownet"
    # worker.quantized_model_forward(X_test[:100].reshape(-1, 28 * 28), Y_test[:100], dump_flag=True, dump_path=dump_path)
    
    # zkfl.generate_mnist_proof()