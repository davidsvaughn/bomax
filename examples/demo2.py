import sys, os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import torch
import logging
from datetime import datetime
from glob import glob
import itertools
import traceback

# Import from the bomax package
from bomax.initialize import init_samples
from bomax.sampler import MultiTaskSampler

from synthetic import generate_learning_curves

torch.set_default_dtype(torch.float64)
rand_seed = -1

#--------------------------------------------------------------------------

def load_example_dataset(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    
    # load data
    df = pd.read_csv(os.path.join(data_dir, file_name), delimiter='\t')

    # Extract checkpoint numbers
    X_steps = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Identify test columns (excluding average)
    task_cols = [col for col in df.columns if col.startswith('TASK_')]
    Y_values = df[task_cols].values
    # n,m = Y_values.shape
    
    return X_steps, Y_values


#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'perf1.txt'
# fn = 'perf2.txt'

# rand_seed = 2951

# number of random obs-per-task (opt) to initially sample
# if <1, then fraction of total obs
n_obs    = 2
# n_obs    = 0.1

# stop iterating after this fraction of all points have been sampled
max_sample = 0.1

#-------------------------------------------------------------------------
# random seed
rand_seed = np.random.randint(1000, 10000) if rand_seed <= 0 else rand_seed
np.random.seed(rand_seed)
random.seed(rand_seed)

# detect if running on local machine
local = os.path.exists('/home/david')

# if not local:
plt.ioff()

#--------------------------------------------------------------------------
# setup run directory

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
run_base = os.path.join(current_dir, 'runs') if local else '/mnt/llm-train/baso/runs'

# Create runs directory if it doesn't exist
os.makedirs(run_base, exist_ok=True)

# remove all empty run directories
for d in glob(os.path.join(run_base, f'run_*')):
    if len(os.listdir(d)) == 0:
        os.rmdir(d)

# create a new run directory
run_id = f'run_{rand_seed}'
j = len(glob(os.path.join(run_base, f'{run_id}*')))
run_dir = os.path.join(run_base, f'{run_id}_{j}' if j>0 else run_id)
while os.path.exists(run_dir): run_dir += 'a'
os.makedirs(run_dir, exist_ok=False)

#-----------------------------------------------------------------------
# setup logging
log_file = os.path.join(run_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    # format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will print to console too
    ]
)
def log(msg, verbosity=1, verbosity_level=1):
    if verbosity >= verbosity_level:
        logging.getLogger(__name__).info(msg)
    
log('-'*110)
log(f'Run directory: {run_dir}')
log(f'Random seed: {rand_seed}')

#--------------------------------------------------------------------------
# example dataset
X_steps, Y_values = load_example_dataset(fn)

# synthetic dataset
X_steps, Y_values =  generate_learning_curves(50, 25)

n, m = Y_values.shape
log(f'X_steps: {X_steps}')
log(f'Y_values shape: {Y_values.shape}')

#--------------------------------------------------------------------------
# gold standard
Y_ref = None
Y_ref = Y_values.copy()

Y_ref_mean = Y_ref.mean(axis=1)
Y_test_mean = Y_values.mean(axis=1)

#--------------------------------------------------------------------------
# find best checkpoint

best_idx = np.argmax(Y_values.mean(axis=1))
best_y_mean = Y_values.mean(axis=1)[best_idx]
best_checkpoint = X_steps[best_idx]

i = np.argmax(Y_ref_mean)
regression_best_checkpoint = X_steps[i]
regression_y_max = Y_ref_mean[i]

log(f'TRU BEST CHECKPOINT:\t{best_checkpoint}\tY={best_y_mean:.4f}')
log(f'REF BEST CHECKPOINT:\t{regression_best_checkpoint}\tY={regression_y_max:.4f}')


#---------------------------------------------------------
# Subsample data
S = init_samples(n, m, min_obs=2, log=log)
Y_obs = np.full(S.shape, np.nan)

# Y_obs[S] = Y_values[S]
for i, j in itertools.product(range(n), range(m)):
        if S[i, j]:
            Y_obs[i, j] = Y_values[i, j]
    
# Initialize the sampler
sampler = MultiTaskSampler(X_steps, Y_obs,
                        #    Y_test=Y_values,
                            max_sample=max_sample, 
                            run_dir=run_dir,
                            eta=0.25,
                            # log_interval=5,
                            )

# Fit model to initial samples
sampler.update()
sampler.compare(Y_ref, Y_values)
# sampler.compare(Y_test)

# Run Bayesian optimization loop
while sampler.sample_fraction < max_sample:
    _, next_task = sampler.add_next_sample(lambda i,j: Y_values[i,j])
    sampler.update()
    sampler.compare(Y_ref, Y_values)
    # sampler.compare(Y_test)
    sampler.plot_task(next_task, '- AFTER')
    # sampler.plot_posterior_mean(y_gold=Y_test_mean)
    sampler.plot_posterior_mean(Y_ref_mean, Y_test_mean)

