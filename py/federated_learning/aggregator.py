import torch
import numpy as np

class FedAvg:
    def __init__(self, lr=1):
        """ 
        Constructor for the `FedAvg` class.
        :param global_model: Dictionary of numpy arrays representing the global model.
        """
        self.lr = lr
    
    def aggregate(self, local_updates):
        """ 
        Method to aggregate the local updates from the workers.
        :param local_updates: List of dictionaries of numpy arrays representing the local updates.
        """
        num_workers = len(local_updates)
        aggregated_updates = {}
        for key in local_updates[0].keys():
            assert all(key in worker for worker in local_updates), f"Key {key} not found in all workers"
            updates = [worker[key] for worker in local_updates]
            aggregated_updates[key] = np.mean(updates, axis=0)
            # aggregated_updates[key] = - self.lr * aggregated_updates[key]
        return aggregated_updates