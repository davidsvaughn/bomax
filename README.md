# BOMAX: Bayesian Optimization using Multi-task Gaussian Process Regression

*Find the LLM checkpoint with the best average performance across multiple benchmarks*

BOMAX is a Python package for Bayesian Optimization with Multi-task Gaussian Process Regression, designed for efficient optimization of expensive-to-evaluate functions across multiple related tasks.

The problem that inspired BOMAX was optimizing LLM learning curves. A learning curve visualizes the performance of a model during training, showing how the performance changes over some time unit (steps/iterations/epochs). In classic ML, the performance is usually measured on a small validation set, sometimes called a *hold-out* set (i.e. 10-20% of the training data is *held out*). The goal is to capture the model parameters at the peak of this curve. However, if we are training/fine-tuning an LLM for multiple uses, our validation set might actually be a combination of many different benchmark tasks. After all, most LLM-leaderbaords rank LLMs by their *average benchmark score*.

Let's say we have a set of model 100 model checkpoints that were saved at regular intervals during training/fine-tuning. Suppose that running a single model on a single benchmark takes 1 minute, and we have 100 benchmark tasks. Running all models on all tasks would take 10000 minutes = 1 week. How could we efficiently estimate the model checkpoint with the highest average benchmark performance without running all model checkpoints through all benchmark tasks?

The key is to use Bayesian Optimization with Multi-task Gaussian Process Regression. 

See here for the full derivation. XXX


## Features

- Uses BoTorch implementation of Multi-task Gaussian Process
- Bayesian optimization with modified Expected Improvement acquisition function (for optimizing an average)
- Efficient sampling strategies for multi-task settings
- Visualization tools for monitoring optimization progress
- Support for both CPU and GPU acceleration

## Installation

You can install BOMAX in development mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/bomax.git
cd bomax

# Install in editable mode
pip install -e .
```

## Requirements

BOMAX requires the following packages:

- numpy
- pandas
- matplotlib
- torch
- gpytorch
- botorch
- scipy
- scikit-learn

These dependencies will be automatically installed when you install the package.

## Usage

Here's a simple example of how to use BOMAX:

```python
import numpy as np
from bomax.initialize import init_samples
from bomax.sampler import MultiTaskSampler

# Define your feature space and observations
X_feats = np.array([...])  # Your input features
Y_obs = np.array([...])    # Your observations (with np.nan for unobserved points)

# Initialize the sampler
sampler = MultiTaskSampler(
    X_feats, 
    Y_obs,
    eta=0.25,              # LKJ prior parameter
    rank_fraction=0.5,     # Rank for low-rank approximation
    ei_beta=0.5,           # Expected Improvement parameter
    use_cuda=True          # Use GPU if available
)

# Fit the model to initial samples
sampler.update()

# Run Bayesian optimization loop
for _ in range(10):
    # Get next sample point
    next_i, next_j = sampler.add_next_sample()
    
    # Update the model
    sampler.update()
    
    # Visualize results
    sampler.plot_task(next_j, '- AFTER')
    sampler.plot_posterior_mean()
```

For a more detailed example, see the `examples/demo.py` file.

## Structure

The package is organized as follows:

- `src/bomax/`: Main package directory
  - `__init__.py`: Package initialization
  - `sampler.py`: MultiTaskSampler class for Bayesian optimization
  - `initialize.py`: Functions for initializing samples
  - `normalize.py`: Data normalization utilities
  - `degree.py`: Degree metric for curve complexity
  - `stopping.py`: Early stopping conditions
  - `utils.py`: Utility functions
- `examples/`: Example scripts
- `data/`: Example datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.
