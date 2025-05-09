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
from bomax.sampler import MultiTaskSampler
# from bomax.initialize import init_samples/

from synthetic import generate_learning_curves

torch.set_default_dtype(torch.float64)
plt.ioff()

# detect if running on local machine
local = os.path.exists('/home/david')

rand_seed = -1
# rand_seed = 8227

#--------------------------------------------------------------------------

def load_example_dataset(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    
    # load data
    df = pd.read_csv(os.path.join(data_dir, file_name), delimiter='\t')

    # Extract checkpoint numbers
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Identify task columns (excluding average)
    task_cols = [col for col in df.columns if col.startswith('TASK_')]
    Y_values = df[task_cols].values
    
    return X_feats, Y_values

#--------------------------------------------------------------------------

# random seed
rand_seed = np.random.randint(1000, 10000) if rand_seed <= 0 else rand_seed
np.random.seed(rand_seed)
random.seed(rand_seed)

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
def log(msg):
    logging.getLogger(__name__).info(msg)
    
log('-'*110)
log(f'Run directory: {run_dir}')
log(f'Random seed: {rand_seed}')

#--------------------------------------------------------------------------

data_file = 'dataset1.txt'
# data_file = 'dataset2.txt'

# example dataset
X_feats, Y_curves = load_example_dataset(data_file)

# synthetic dataset
# X_feats, Y_curves =  generate_learning_curves(50, 50)

num_inputs, num_outputs = Y_curves.shape
log(f'Y_curves.shape={Y_curves.shape} (num_inputs={num_inputs} checkpoints, num_outputs={num_outputs} tasks)')

#--------------------------------------------------------------------------
# Compute Gold Standard Variants

# Raw (noisy) mean over all tasks (i.e. all learning curves) 
Y_mean = Y_curves.mean(axis=1)

# Smooth each task (learning curve) independently
Y_smooth_curves = np.array([ndimage.gaussian_filter1d(col, sigma=num_inputs/15) for col in Y_curves.T]).T

# Smoothed mean = mean of all smoothed curves
Y_smooth_mean = Y_smooth_curves.mean(axis=1)

#--------------------------------------------------------------------------
# Compute True Optimal Checkpoint (using Gold Standard data)

# using raw data
i = np.argmax(Y_curves.mean(axis=1))
raw_x_max, raw_y_max = X_feats[i], Y_curves.mean(axis=1)[i]

# using smoothed data
i = np.argmax(Y_smooth_mean)
smooth_x_max, smooth_y_max = X_feats[i], Y_smooth_mean[i]

log(f'BEST CHECKPOINT:')
log(f'   RAW:     \t{raw_x_max}\tY={raw_y_max:.4f}')
log(f'   SMOOTHED:\t{smooth_x_max}\tY={smooth_y_max:.4f}')

#--------------------------------------------------------------------------

# Initialize the Sampler
sampler = MultiTaskSampler(num_inputs, num_outputs,
                           func=lambda i,j: Y_curves[i,j],  # black-box function callback
                           X_feats=X_feats, # optional (important when not equally spaced)
                           run_dir=run_dir)

# seed with initial observations (at least 2 obs/task) for numerical stability
sampler.initialize()

# Fit model to initial observations
sampler.update()

# Compare with Gold Standard data
sampler.compare(Y_smooth_curves)

#---------------------------------------------------------------------------

# Main Bayesian optimization Loop
while sampler.sample_fraction < 0.1:
    
    # determine next sample coordinates, and query black-box function
    next_checkpoint, next_task = sampler.sample()
    
    # update the GP model with the new sample
    sampler.update()
    
    # Compare with Gold Standard data and plot results
    sampler.compare(Y_smooth_curves)
    # sampler.plot_task(next_task, '- AFTER')
    sampler.plot_posterior_mean(Y_smooth_mean, Y_mean)

