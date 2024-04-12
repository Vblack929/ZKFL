import torch
import numpy as np

class FedAvg:
    def __init__(self, global_model):
        """ 
        Constructor for the `FedAvg` class.
        :param global_model: Dictionary of numpy arrays representing the global model.
        """
        self.global_model = global_model
        self.params = self.global_model.get_params()
        
    def aggregate(self, local_params):
        """ 
        Method to aggregate the local updates from the workers.
        :param local_updates: List of dictionaries of numpy arrays representing the local updates.
        
        Returns:
        new_params: List of numpy arrays representing the new global model parameters.
        """
        self.local_params = local_params
        
        round_agg = [np.zeros_like(p) for p in self.global_model.get_params()]
        for param in self.local_params:
            for i, p in enumerate(param):
                round_agg[i] += p
        N = len(self.local_params)
        round_agg = [p / N for p in round_agg]
        
        # for i, p in enumerate(self.params):
        #     self.params[i] += self.lr * round_agg[i]
        # self.global_model.set_params(self.params)
        # self.global_model.set_params(round_agg)
        return round_agg

class GuassianAttack:
    """ 
    Malicious aggregation that set the global model with random numbers that follow a Gaussian distribution.
    """
    def __init__(self, global_model):
        self.global_model = global_model
        self.params = self.global_model.get_params()    
        
    def aggregate(self, local_params):
        """Method to attack the aggregation process by setting the global model with random numbers that follow a Gaussian distribution.

        Args:
            local_params : List of dictionaries of numpy arrays representing the local updates.
        Returns:
            new_params : List of numpy arrays of the same shape as the global model parameters but with random values.
        """
        self.local_params = local_params
        
        new_params = [np.random.normal(size=p.shape) for p in self.params]  
        return new_params
        