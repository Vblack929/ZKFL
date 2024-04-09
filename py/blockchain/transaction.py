import time
import hashlib

class Transaction:
    def __init__(self, sender_id: str, task: str, params: list, accuracy: float, timestamp: float, verified: bool = False):
        """ 
        Constructor for the `Transaction` class.
        :param sender_id: Unique ID of the sender.
        :param task: Task to be performed.
        :param params: Parameters of the model to be used.
        :param timestamp: Time of generation of the transaction.
        """
        self.sender_id = sender_id
        self.task = task
        self.params = params
        self.accuracy = accuracy
        self.timestamp = timestamp
        self.verified = verified
        self.transaction_hash = self.compute_hash()
        
    def compute_hash(self):
        """ 
        Returns the hash of the transaction instance by first converting it into JSON string.
        """
        transaction_string = "{}{}{}{}".format(self.sender_id, self.task, self.params, self.timestamp)
        return hashlib.sha256(transaction_string.encode()).hexdigest()
    
    def __repr__(self):
        return f"Transaction <Sender: {self.sender_id}, Task: {self.task}, Accuracy: {self.accuracy}, Hash: {self.transaction_hash}>"