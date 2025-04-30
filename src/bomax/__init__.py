"""
bomax - Bayesian Optimization for maximizing multiple learning curves
"""

__version__ = "0.1.0"

from . import degree
from . import initialize
from . import normalize
from . import sampler
from . import stopping
from . import utils

# Expose key classes and functions for easier imports
from .normalize import Transform
from .sampler import MultiTaskSampler
from .initialize import init_samples
