import requests
import time
import copy
from blockchain.blockchain import Block, Blockchain, Miner
from blockchain.transaction import Transaction
from federated_learning.model.simpleCNN import SimpleCNN
from federated_learning.client import Worker
from federated_learning.data.mnist_processing import get_mnist_dataloaders
from federated_learning.aggregator import FedAvg
from federated_learning.utils import numpy_dict_to_model, model_to_numpy_dict
from server import PoWThread
import threading

blockchain_lock = threading.Lock()


def powNetwork(model_struct, num_workers, global_epochs, local_epochs, task):
    blockchain = Blockchain(
        model_struct=copy.deepcopy(model_struct), task=task)

    # create workers
    workers = []
    for i in range(num_workers):
        # get the base model from the genesis block
        worker = Worker(
            index=i + 1,
            dataset=get_mnist_dataloaders(train_fraction=0.2),
            model_struct=copy.deepcopy(model_struct)
        )
        workers.append(worker)

    public_test_data = get_mnist_dataloaders(train_fraction=0.1)['test']

    # federated learning training
    print("Starting Federated Learning Training")
    for epoch in range(global_epochs):
        print(f"Epoch {epoch+1}/{global_epochs}")
        base_params = blockchain.last_block.global_params
        base_model = numpy_dict_to_model(base_params, model_struct)
        for worker in workers:
            # get the global model from the last block
            if blockchain.last_block.global_params is not None:
                worker.get_model(model_params=base_model.state_dict())
            else:
                raise ValueError("Global model not found")
            worker.local_train(epochs=local_epochs)
            local_update = worker.local_update(eval=True)
            # send local update as transaction to the pool
            tx = Transaction(
                sender_id=worker.index,
                task=task,
                model_params=local_update['model'],
                accuracy=local_update['accuracy'],
                timestamp=time.time(),
                verified=False
            )
            blockchain.add_transaction(tx)
            print(f"Worker {worker.index} sent transaction")
        # mine block using pow
        print("Local updates sent. Mining block...")
        stop_event = threading.Event()
        miners = [Miner(blockchain=blockchain, miner_id=i+1,
                        stop_event=stop_event) for i in range(num_workers)]

        for miner in miners:
            miner.start()

        for miner in miners:
            miner.join()

        # aggregation by the winner
        leader_idx = blockchain.last_block.miner_id
        leader = workers[leader_idx-1]
        print(f"Leader: worker {leader_idx}")

        # leader test the global model
        leader.get_model(model_params=base_model.state_dict())
        acc = leader.evaluate(public_test_data)
        print(f"Gloabl Accuracy: {acc}%")

        if not blockchain.valid_chain():
            print("Chain is invalid. Stopping training")
            break
        
def polNetwork(model_struct, num_workers, global_epochs, local_epochs, task):
    blockchain = Blockchain(
        consensus="PoL",
        model_struct=copy.deepcopy(model_struct),
        task=task
    )
    # create nodes
    workers = []
    for i in range(num_workers):
        worker = Worker(
            index=i + 1,
            dataset=get_mnist_dataloaders(train_fraction=0.2),
            model_struct=copy.deepcopy(model_struct)
        )
        workers.append(worker)
    public_test_data = get_mnist_dataloaders(train_fraction=0.1)['test']
    # federated learning training
    print("Starting Federated Learning Training")
    for epoch in range(global_epochs):
        print(f"Epoch {epoch+1}/{global_epochs}")
        base_params = blockchain.last_block.global_params
        base_model = numpy_dict_to_model(base_params, model_struct)
        for worker in workers:
            # get the global model from the last block
            if blockchain.last_block.global_params is not None:
                worker.get_model(model_params=base_model.state_dict())
            else:
                raise ValueError("Global model not found")
            worker.local_train(epochs=local_epochs)
            acc = worker.evaluate(data=public_test_data)
            local_update = worker.local_update(eval=True)
            # send local update as transaction to the pool
            tx = Transaction(
                sender_id=worker.index,
                task=task,
                model_params=local_update['model'],
                accuracy=acc,
                timestamp=time.time(),
                verified=False
            )
            blockchain.add_transaction(tx)
            print(f"Worker {worker.index} sent transaction")
        # mine block using pow
        print("Local updates sent. Mining block...")
        # sort the transactions based on accuracy
        blockchain.sort_transactions()
        # all node participating in verification
        
        
    


if __name__ == '__main__':
    model_struct = SimpleCNN()
    num_workers = 3
    global_epochs = 2
    local_epochs = 2
    task = "mnist"
    powNetwork(model_struct, num_workers, global_epochs, local_epochs, task)
