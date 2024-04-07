import time
import hashlib

class Transaction:
    def __init__(self, sender_id: str, task: str, model, model_params: dict, accuracy: float, timestamp: float, proof, verified: bool = False):
        """ 
        Constructor for the `Transaction` class.
        :param sender_id: Unique ID of the sender.
        :param task: Task to be performed.
        :param model_params: Parameters of the model to be used.
        :param timestamp: Time of generation of the transaction.
        """
        self.sender_id = sender_id
        self.task = task
        self.model = model
        self.model_params = model_params
        self.accuracy = accuracy
        self.timestamp = timestamp
        self.verified = verified
        self.proof = proof
        self.transaction_hash = self.compute_hash()
        
    def compute_hash(self):
        """ 
        Returns the hash of the transaction instance by first converting it into JSON string.
        """
        transaction_string = "{}{}{}{}{}".format(self.sender_id, self.task, self.model_params, self.timestamp, self.verified)
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def __repr__(self):
        return f"Transaction <Sender: {self.sender_id}, Task: {self.task}, Accuracy: {self.accuracy}, Hash: {self.transaction_hash}>"
    
class TransactionPool:
    def __init__(self):
        """ 
        Constructor for the `TransactionPool` class.
        """
        self.transactions = []
        
    @property
    def size(self):
        """ 
        Property to return the number of transactions in the pool.
        """
        return len(self.transactions)
        
    def add_transaction(self, transaction: Transaction):
        """ 
        Method to add a transaction to the pool.
        :param transaction: Transaction to be added to the pool.
        """
        if not isinstance(transaction, Transaction):
            raise ValueError("Invalid transaction object")
        self.transactions.append(transaction)
        
    def remove_transaction(self, transaction: Transaction):
        """ 
        Method to remove a transaction from the pool.
        :param transaction: Transaction to be removed from the pool.
        """
        self.transactions.remove(transaction)
        
    def get_transactions(self):
        """ 
        Method to return all the transactions in the pool.
        """
        return self.transactions