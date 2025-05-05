import numpy as np
import matplotlib.pyplot as plt
import os, sys
import itertools
from math import ceil
import random

#--------------------------------------------------------------

def init_samples(N, M, min_obs=2, n_obs=None, log=None):
    """
    Initialize samples for N x_indices and M tasks.
    
    Parameters:
    N (int): Number of x_indices
    M (int): Number of tasks
    n_obs (float):  if >=1, Initial number of observations (i.e. x_indices) per task
                if <1, Initial fraction of full sample space (KxZ)
    log (function, optional): Logging function to print messages
    """
    if n_obs is None:
        n_obs = min_obs
        
    if n_obs >= 1:
        if n_obs < min_obs:
            n_obs = min_obs
            if log: log('FYI: increasing n_obs to min_obs (minimum 2 obs/task allowed)')
        m = ceil(n_obs * M)
    else:
        min_frac = min_obs/N # == 2*M/(N*M)
        if n_obs < min_frac:
            m = min_obs*M
            if log: log(f'FYI: increasing n_obs to {min_frac:.4g} (minimum 2 obs/task allowed)')
        else:
            m = max(min_obs*M, ceil(n_obs * N * M))
    if log: log(f'FYI: initializing sampler with {m} observations ( ~{m/(N*M):.4g} of all obs, ~{m/M:.4g} obs/task )')
    if log: log('-'*80)
    
    tasks = list(range(M))
    x_indices = list(range(N))
    x_tasks = [[] for _ in range(N)]
    n = 0
    while True:
        # select a random x_index
        random.shuffle(x_indices)
        for x_idx in x_indices:
            sub_tasks = [tt for tt in tasks if tt not in x_tasks[x_idx]]
            if len(sub_tasks) > 0:
                break
        t = random.choice(sub_tasks)
        x_tasks[x_idx].append(t)
        n += 1
        if n >= m:
            break
        tasks.remove(t)
        x_indices.remove(x_idx)
        if len(tasks) == 0:
            tasks = list(range(M)) # reset task list
        if len(x_indices) == 0:
            x_indices = list(range(N))
    random.shuffle(x_tasks)
    
    # convert to x,y indices, and create a boolean matrix of size (N, M)
    S = np.zeros((N, M), dtype=bool)
    for i, tasks in enumerate(x_tasks):
        for j in tasks:
            S[i, j] = True
    
    # use: X,Y = np.where(S)
    return S


#--------------------------------------------------------------
if __name__ == "__main__":

    N,M = 120, 70
    n_obs = 2

    S = init_samples(N, M, n_obs)
    # X,Y = np.where(S)
    
    print(S.sum(axis=0))
    print(S.sum(axis=1))
    print(S.sum())
