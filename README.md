# BOMAX: Bayesian Optimization using Multi-task Gaussian Process Regression

BOMAX is a Python package for Bayesian Optimization with Multi-task Gaussian Process Regression, designed for efficient optimization of expensive-to-evaluate functions across multiple related tasks, such as evaluating LLM checkpoints across multiple benchmark tasks in order to optimize the average learning curve (i.e. *find the LLM checkpoint with the best average performance across all benchmarks*).

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
