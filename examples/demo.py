import sys, os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import torch
import logging
from datetime import datetime
from glob import glob
import traceback

# Import from the bomax package
from bomax.initialize import init_samples
from bomax.sampler import MultiTaskSampler

torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_fraction = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'perf1.txt'
# fn = 'perf2.txt'

# rand_seed = 2951

# number of random obs-per-task (opt) to initially sample
# if <1, then fraction of total obs
n_obs    = 2
# n_obs    = 0.1

# stop BO after this fraction of all points are sampled
# if <1, then fraction of total obs
# if >1, then number of obs
max_sample = 600

# select random subset of tasks
task_sample = 1.0
# task_sample = 0.35

# multi-task lkj prior
eta = 0.25
eta_gamma = 0.99
rank_fraction = 0.5 # 0.25  0.5

# Expected Improvement parameters...
ei_beta = 0.5
# deta decay: ei_beta will be ei_f of its start value (i.e. 0.1) when ei_t of all points have been sampled
ei_f, ei_t = 0.2, 0.05
# ei_gamma = 0.9925

# misc
log_interval = 25
verbosity = 1
use_cuda = True

# synthetic data...
# synthetic = False
# n_rows, n_cols = 100, 100

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
# Load the Data...

# We'll assume you have a CSV with columns:
# "CHECKPOINT", "TASK_1", "TASK_2", ....
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
run_base = os.path.join(current_dir, 'runs') if local else '/mnt/llm-train/baso/runs'

# Create runs directory if it doesn't exist
os.makedirs(run_base, exist_ok=True)

#--------------------------------------------------------------------------
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

# copy current file to run directory (save parameters, etc)
src = os.path.join(current_dir, os.path.basename(__file__))
dst = os.path.join(run_dir, os.path.basename(__file__))
os.system(f'cp {src} {dst}')

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
def log(msg, verbosity_level=1):
    if verbosity >= verbosity_level:
        logging.getLogger(__name__).info(msg)
    
log('-'*110)
log(f'Run directory: {run_dir}')
log(f'Random seed: {rand_seed}')

#--------------------------------------------------------------------------
# load data
df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')

# Extract checkpoint numbers
X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
 
# Identify test columns (excluding average)
task_cols = [col for col in df.columns if col.startswith('TASK_')]
Y_test = df[task_cols].values
del task_cols
n,m = Y_test.shape

# sample subset of tasks (possibly)
if task_sample>0 and task_sample!=1:
    if task_sample < 1:
        task_sample = int(task_sample * m)
    idx = np.random.choice(range(m), task_sample, replace=False)
    Y_test = Y_test[:, idx]
    n,m = Y_test.shape
    
if max_sample > 1:
    max_sample = max_sample / (n*m)
    
# compute ei_gamma
ei_gamma = np.exp(np.log(ei_f) / (ei_t*n*m - 2*m))
log(f'FYI: ei_gamma: {ei_gamma:.4g}')

#--------------------------------------------------------------------------
# Train regression model on all data for gold standard
Y_ref = None
Y_ref = Y_test.copy()

if Y_ref is None:
    sampler = MultiTaskSampler(X_feats, Y_test, 
                               Y_test=Y_test,
                               eta=None,
                               degree_thresh=6,
                               min_iterations=100,
                               max_iterations=2000,
                               loss_thresh=0.00025,
                               log_interval=25,
                               use_cuda=use_cuda,
                               run_dir=run_dir,
                               )
    # Fit model to full dataset
    _, _, Y_ref, _ = sampler.update()
    
    sampler.compare(Y_test)

Y_ref_mean = Y_ref.mean(axis=1)
Y_test_mean = Y_test.mean(axis=1)

#--------------------------------------------------------------------------
# find best checkpoint

best_idx = np.argmax(Y_test.mean(axis=1))
best_y_mean = Y_test.mean(axis=1)[best_idx]
best_checkpoint = X_feats[best_idx]

i = np.argmax(Y_ref_mean)
regression_best_checkpoint = X_feats[i]
regression_y_max = Y_ref_mean[i]

log(f'TRU BEST CHECKPOINT:\t{best_checkpoint}\tY={best_y_mean:.4f}')
log(f'REF BEST CHECKPOINT:\t{regression_best_checkpoint}\tY={regression_y_max:.4f}')

#--------------------------------------------------------------------------
# run the sampler

for _ in range(10): # try 10 times to complete the run without error
    
    try:
        # Subsample data
        S = init_samples(n, m, n_obs, log=log)
        Y_obs = np.full(S.shape, np.nan)
        Y_obs[S] = Y_test[S]
            
        # Initialize the sampler
        sampler = MultiTaskSampler(X_feats, Y_obs,
                                   Y_test=Y_test,
                                   eta=eta,
                                   eta_gamma=eta_gamma,
                                   ei_beta=ei_beta,
                                   ei_gamma=ei_gamma,
                                   max_sample=max_sample, 
                                   rank_fraction=rank_fraction,
                                   log_interval=log_interval,
                                   use_cuda=use_cuda,
                                   run_dir=run_dir,
                                   max_retries=20,
                                   )

        # Fit model to initial samples
        sampler.update()
        sampler.compare(Y_ref, Y_test)
        # sampler.compare(Y_test)
        # sampler.plot_all(max_fig=10)

        # Run Bayesian optimization loop
        while sampler.sample_fraction < max_sample:
            _, next_task = sampler.add_next_sample()
            sampler.update()
            sampler.compare(Y_ref, Y_test)
            # sampler.compare(Y_test)
            sampler.plot_task(next_task, '- AFTER')
            # sampler.plot_posterior_mean(y_gold=Y_test_mean)
            sampler.plot_posterior_mean(Y_ref_mean, Y_test_mean)
            
        break
        
    except Exception as e:
        logging.error(f'ERROR: {e}')
        logging.error(traceback.format_exc())
        pass

#--------------------------------------------------------------------------
