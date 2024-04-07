import hashlib
import time
import json
import copy
import requests
import os
from urllib.parse import urlparse
from typing import List
import threading
from .transaction import Transaction, TransactionPool
from federated_learning.aggregator import FedAvg
from federated_learning.utils import numpy_dict_to_model, model_to_numpy_dict


class Block:
    def __init__(self, index: int, transactions: List[Transaction], timestamp: float, previous_hash: str, global_params: dict = None):
        """ 
        Constructor for the `Block` class.
        :param index: Unique ID of the block.
        :param transactions: List of transactions.
        :param timestamp: Time of generation of the block.
        :param previous_hash: Hash of the previous block in the chain which this block is part of.
        """
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = 0
        self.miner_id = 0
        self.global_params = global_params
        self.global_accuracy = 0.0
        self.mining_time = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        """ 
        Returns the hash of the block instance by first converting it into JSON string.
        """
        block_string = "{}{}{}{}{}".format(
            self.index, self.transactions, self.timestamp, self.previous_hash, self.nonce)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __repr__(self):
        return f"Block <Index: {self.index}, Hash: {self.hash}, Previous Hash: {self.previous_hash}>"

    def __str__(self):
        return {
            "index": self.index,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "transactions": [str(transaction) for transaction in self.transactions],
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "global accuracy": self.global_accuracy,
            "miner_id": self.miner_id,
            "mining_time": self.mining_time,
        }


class Blockchain:
    def __init__(self, consensus="pow", max_nodes=10, model_struct=None, task='mnist', save_path="blockchain/data"):
        """ 
        Constructor for the `Blockchain` class. 
        """
        self.chain = []
        self.consensus = consensus
        self.max_nodes = max_nodes
        self.model_struct = model_struct
        self.task = task
        self.save_path = save_path
        self.peers = set()
        self.transaction_pool = []
        self.genesis_block = self.create_genesis_block()
        self.add_block(self.genesis_block)
        self.store_block(self.genesis_block)

    def create_genesis_block(self):
        """ 
        Method to create the first block in the chain.
        """
        # convert base model to transaction
        global_model = self.model_struct
        first_tx = Transaction(
            sender_id="0",
            task=self.task,
            model=global_model,
            model_params=None,
            accuracy=0.0,
            timestamp=time.time()
        )
        return Block(
            index=1,
            transactions=[first_tx],
            timestamp=time.time(),
            previous_hash="0",
            global_params=model_to_numpy_dict(global_model)
        )

    def register_client(self, client):
        self.peers.add(client)

    def make_block(self, miner, previous_hash=None, transactions=None):
        """ 
        Method to create a new block in the chain.
        :param previous_hash: Hash of the previous block in the chain.
        """
        if previous_hash is None:
            previous_hash = self.last_block.hash

        if transactions is None:
            verified_tx = []
            for tx in self.transaction_pool:
                if miner.verify(tx):
                    verified_tx.append(tx)
        index = len(self.chain) + 1
        block = Block(
            index=index,
            transactions=verified_tx,
            timestamp=time.time(),
            previous_hash=previous_hash,
        )
        block.miner_id = miner.miner_id
        return block

    def store_block(self, block, dump_model=False):
        """ 
        Method to store a block in the chain.
        :param block: Block to be stored in the chain.
        """
        file_name = "block" + str(block.index) + ".txt"
        file_path = os.path.join(self.save_path, file_name)

        with open(file_path, "w") as f:
            f.write(json.dumps(block.__str__(), indent=4))
            f.write("\n")

        if dump_model:
            if block.global_params is not None:
                model_params = {}
                for key, value in block.global_params.items():
                    model_params[key] = value.tolist()
                with open(file_path, "w") as f:
                    f.write(json.dumps(model_params, indent=4))
                    f.write("\n")
        print(f"Block {block.index} stored.")

    @staticmethod
    def hash(data):
        return hashlib.sha256(data.encode()).hexdigest()

    @property
    def last_block(self):
        """ 
        Method to return the last block in the chain.
        """
        return self.chain[-1]

    def __len__(self):
        return len(self.chain)

    @staticmethod
    def valid_proof(block, difficulty=5):
        guess_hash = block.compute_hash()
        return guess_hash[:difficulty] == '0'*difficulty

    def add_transaction(self, transaction: Transaction):
        self.transaction_pool.append(transaction)
        with open("blockchain/data/transaction_pool.txt", "a") as f:
            f.write(transaction.__repr__() + "\n")

    def add_block(self, block):
        self.chain.append(block)

    def aggregate_models(self, block):
        aggregator = FedAvg()
        local_params = []
        for tx in block.transactions:
            local_params.append(tx.model_params)

        aggregated_params = aggregator.aggregate(local_params)
        block.global_params = aggregated_params
        print("Local models aggregated.")
        
    def sort_transactions(self):
        """ 
        Sort the transactions in the pool based on the accuracy of the model.
        """
        self.transaction_pool = sorted(
            self.transaction_pool, key=lambda x: x.accuracy, reverse=True)
    
    def remove_transactions(self, tx):
        """ 
        Remove a transaction from the transaction pool.
        """
        self.transaction_pool = [t for t in self.transaction_pool if t != tx]
    
    def empty_transaction_pool(self):
        """ 
        Empty the transaction pool.
        """
        self.transaction_pool = []

    def proof_of_work(self, block, stop_event):
        stopped = False
        while not self.valid_proof(block):
            if stop_event.is_set():
                stopped = True
                break
            block.nonce += 1
            # if block.nonce % 1000 == 0:
            #     print(f"Block {block.index} mining, nonce: {block.nonce}")
        if not stopped:
            print(f"Block {block.index} mined by {block.miner_id}")
            # remove verified transactions from the pool
            self.transaction_pool = [
                tx for tx in self.transaction_pool if tx not in block.transactions]
            # update the transaction_pool file
            with open("blockchain/data/transaction_pool.txt", "w") as f:
                for tx in self.transaction_pool:
                    f.write(tx.__repr__() + "\n")
        if stopped:
            print("mining stopped")
        else:
            print("done")
        return block

    def valid_chain(self):
        last_block = self.chain[0]
        cur_index = 1
        print(len(self.chain))
        while cur_index < len(self.chain):
            block = self.chain[cur_index]
            if block.previous_hash != last_block.hash:
                print("previous hash mismatch", cur_index)
                print(block.previous_hash, last_block.hash)
                return False
            if self.consensus == "pow":
                if not self.valid_proof(block):
                    print("invalid proof", cur_index)
                    return False
            last_block = block
            cur_index += 1
        return True

    def resolve_conflicts(self, stop_event):
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.chain)
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
            if new_chain:
                stop_event.set()
                self.chain = new_chain
                block = self.last_block()
                resp = requests.post(
                    f"http://{bnode}/new_block", json={"block": block.__repr__()})
                if resp.status_code == 200:
                    print(f"New block sent to {bnode}")
                else:
                    print(f"Failed to send new block to {bnode}")
                return True
        return False


blockchain_lock = threading.Lock()
THRESHOLD = 0.2


class Miner(threading.Thread):
    def __init__(self, miner_id, blockchain, stop_event):
        super().__init__()
        self.miner_id = miner_id
        self.blockchain = blockchain
        self.stop_event = stop_event

    def verify(self, transaction):
        if transaction.accuracy > THRESHOLD:
            return True
        return False

    def run(self):
        while not self.stop_event.is_set():
            # mine block
            start_time = time.time()
            new_block = self.blockchain.make_block(miner=self)
            proof_of_work_res = self.blockchain.proof_of_work(
                new_block, self.stop_event)
            if proof_of_work_res and not self.stop_event.is_set():
                with blockchain_lock:
                    if self.blockchain.valid_proof(new_block):
                        self.blockchain.add_block(new_block)
                        print(
                            f"Miner {self.miner_id} mined block {new_block.index}")
                        self.stop_event.set()
                        end_time = time.time()
                        mining_time = end_time - start_time
                        new_block.mining_time = mining_time
                        self.blockchain.aggregate_models(new_block)
                        self.blockchain.store_block(new_block, dump_model=False)
                        break
                    
class MinerPoL(threading.Thread):
    def __init__(self, miner_id, blockchain, stop_event):
        super().__init__()
        self.miner_id = miner_id
        self.blockchain = blockchain
        self.stop_event = stop_event
