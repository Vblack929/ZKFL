import torch
import numpy as np

class FedAvg:
    def __init__(self, global_model, beta, lr):
        """ 
        Constructor for the `FedAvg` class.
        :param global_model: Dictionary of numpy arrays representing the global model.
        """
        self.global_model = global_model
        self.params = self.global_model.get_params()
        self.beta = beta
        self.lr = lr
        
    def aggregate(self, local_params):
        """ 
        Method to aggregate the local updates from the workers.
        :param local_updates: List of dictionaries of numpy arrays representing the local updates.
        """
        self.local_params = local_params
        
        round_agg = [np.zeros_like(p) for p in self.global_model.get_params()]
        for param in self.local_params:
            for i, p in enumerate(param):
                round_agg[i] += p
        N = len(self.local_params)
        round_agg = [p / N for p in round_agg]
        
        for i, p in enumerate(self.params):
            self.params[i] += self.lr * round_agg[i]
        self.global_model.set_params(self.params)
        return self.global_model
        