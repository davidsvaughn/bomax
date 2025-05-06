import sys, os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import torch
import logging
from datetime import datetime
from glob import glob
from scipy import ndimage

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
# X_steps, Y_values =  generate_learning_curves(50, 25)

# n == number of steps/checkpoints
# m == number of tasks
n, m = Y_values.shape
# log(f'X_steps: {X_steps}')
log(f'Y_values shape: {Y_values.shape}')

#--------------------------------------------------------------------------
# Compute Gold Standard Variants

# Raw (noisy) mean over all tasks (full dataset) 
Y_mean = Y_values.mean(axis=1)

# Smooth each task independently
Y_smooth = np.array([ndimage.gaussian_filter1d(col, sigma=n/10) for col in Y_values.T]).T

# Smoothed mean over all tasks
Y_smooth_mean = Y_smooth.mean(axis=1)

#--------------------------------------------------------------------------
# Find Optimal Checkpoint (using gold standard data)

# raw data
i = np.argmax(Y_values.mean(axis=1))
raw_x_max, raw_y_max = X_steps[i], Y_values.mean(axis=1)[i]

# smoothed data
i = np.argmax(Y_smooth_mean)
smooth_x_max, smooth_y_max = X_steps[i], Y_smooth_mean[i]

log(f'BEST CHECKPOINT:')
log(f'\tRAW:     \t{raw_x_max}\tY={raw_y_max:.4f}')
log(f'\tSMOOTHED:\t{smooth_x_max}\tY={smooth_y_max:.4f}')

#--------------------------------------------------------------------------
# Training Steps

# Get boolean mask S for initial sample points...
# **NOTE** : We need 2 samples-per-task to avoid numerical instability
S = init_samples(n, m, log=log)

# Get initial samples according to boolean mask S
Y_obs = np.where(S, Y_values, np.nan)

# Define sampler and seed with initial samples            
sampler = MultiTaskSampler(X_steps, Y_obs, 
                           eta=0.25,
                        #    use_cuda=True,
                           run_dir=run_dir)

# Fit model to initial samples
sampler.update()

# Compare with Gold Standard data (only possible in test runs)
sampler.compare(Y_smooth, Y_values)

# Run Bayesian optimization loop
while sampler.sample_fraction < 0.15:
    
    # determine next sample coordinates and query black-box function - takes optional callback function: f(i,j)
    _, next_task = sampler.add_next_sample(lambda i,j: Y_values[i,j])
    
    # update the GP model with the new sample
    sampler.update()
    
    # Compare with Gold Standard data, and plot results
    sampler.compare(Y_smooth, Y_values)
    # sampler.plot_task(next_task, '- AFTER')
    sampler.plot_posterior_mean(Y_smooth_mean, Y_mean)

