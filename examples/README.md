# BOMAX Examples

This directory contains example scripts demonstrating how to use the BOMAX package.

## Available Examples

- `demo.py`: A comprehensive example showing how to use the MultiTaskSampler for Bayesian optimization across multiple tasks.

## Running the Examples

To run the examples, make sure you have installed the BOMAX package in development mode:

```bash
# From the root directory of the repository
pip install -e .
```

Then, you can run the examples:

```bash
# From the examples directory
python demo.py
```

## Example Data

The examples use data files from the `data/` directory in the repository root. See the README.md in that directory for more information about the data format.

## Creating Your Own Examples

When creating your own examples, follow these guidelines:

1. Import the necessary modules from the BOMAX package:
   ```python
   from bomax.initialize import init_samples
   from bomax.sampler import MultiTaskSampler
   ```

2. Load your data in the expected format (see the data directory README for details).

3. Initialize the sampler with your data and desired parameters.

4. Run the optimization loop, updating the model and selecting new sample points.

5. Visualize the results using the plotting functions provided by the MultiTaskSampler class.

## Output

The examples will create a `runs/` directory to store logs and visualizations. Each run will have its own subdirectory with a timestamp.
