
### Automatic privacy level computation
from scipy.optimize import fsolve
from scipy.stats import norm
from torch.utils.data import DataLoader
import numpy as np
import math

def fn_dong(mus):
    return math.sqrt(2)*math.sqrt(math.exp(mus*mus)*norm.cdf(1.5*mus)+3*norm.cdf(-0.5*mus)-2)
    
def calc_privacy_lvl(C, tau, T, n, N, d, K):
    #print(C, tau, T, n, N, d, K)
    n_eff = n + (n*n*tau*tau)/(C*C)
    #print("neff=", n_eff)
    #mu_step = (d+(2*n_eff-1)*K)/(n_eff*math.sqrt(2*d + 4*n_eff*K))
    mu_step = (d+(2*n-1)*K)/(n_eff*math.sqrt(2*d + 4*((n*n)/n_eff)*K))
    #print("mu_step=", mu_step)
    c = (n*math.sqrt(T))/N
    mu_tot =  c*fn_dong(mu_step)
    return mu_tot

def calc_dp_privacy_lvl(C, tau, T, n, N, d, K):
    sigma = (n*tau)/(2*C)
    mu_step = 1./sigma
    #print("mu_step=", mu_step)
    c = (n*math.sqrt(T))/N
    mu_tot =  c*fn_dong(mu_step)
    return mu_tot

def get_fn(p_target, dp, C, K, d, N, T, batch_size):
    if dp:
        return (lambda tvar: calc_dp_privacy_lvl(C, tvar, T, batch_size, N, d, K)-p_target)
    else:
        return (lambda tvar: calc_privacy_lvl(C, tvar, T, batch_size, N, d, K)-p_target)

def compute_tau(mu_target, C, K, d, N, T, batch_size, dp=True, init_value=1.0):
    """ Numerically ompute the noise level tau that is required for guaranteeing
        GMIP (dp=False) or GDP (dp=True). 
        For MIP, if the result of the computation yields a higher noise level than DP,
        the DP noise level is used, as DP always implies MIP.
        :param init_value: the initial value to start the numerical search.
    """
    # Rule out cases of MIP, where no noise is required first.
    if dp == False and mu_target > calc_privacy_lvl(C, 0.0, T, batch_size, N, d, K):
        return 0.0

    ## Compute DP level first.
    func_dp = get_fn(mu_target, dp=True, C=C, K=K, d=d, N=N, T=T, batch_size=batch_size)
    tau_0 = np.array([init_value])
    tau_eff_dp = fsolve(func_dp, tau_0.copy(), factor=0.1)[0]

    if dp == False: # Compute noise level for MIP and take min of both
        func_mip = get_fn(mu_target, dp=False, C=C, K=K, d=d, N=N, T=T, batch_size=batch_size)
        tau_0 = np.array([init_value])
        tau_eff = fsolve(func_mip, tau_0.copy(), factor=0.1)[0]
        return min(tau_eff, tau_eff_dp)
    else:
        return tau_eff_dp

from scipy.stats import ncx2, norm
import sys
def analytical_tpr(fpr, mu):
    return 1-norm.cdf(norm.ppf(1-fpr)-mu)


""" A data loader that sequentially passes through a lsit of data loaders """
class ListDataLoader(DataLoader):
    """ A list of data loaders """
    class ListLoaderIter():
        def __init__(self, parent):
            self.parent = parent
            self.curr_idx = 0
            self.curr_iter = iter(parent.my_dllist[0])

        def __next__(self):
            mybatch = None
            while mybatch is None:
                try:
                    mybatch = next(self.curr_iter)
                except StopIteration:
                    if self.curr_idx < len(self.parent.my_dllist) - 1:
                        self.curr_idx +=1
                        self.curr_iter = iter(self.parent.my_dllist[self.curr_idx])
                    else:
                        print("ListDataLoader exhausted")
                        break
            if mybatch is not None:
                return mybatch
            else:
                raise StopIteration         
                    
    def __init__(self, *dls):
        self.my_dllist = dls

    def __iter__(self):
        return ListDataLoader.ListLoaderIter(self)

    def __len__(self):
        return sum([len(myload) for myload in self.my_dllist])
