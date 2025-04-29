# Data Directory

This directory contains example data files for use with the BOMAX package.

## Expected Data Format

The data files should be tab-separated values (TSV) files with the following format:

```
CHECKPOINT	TASK_1	TASK_2	TASK_3	...
checkpoint-1000	0.75	0.68	0.82	...
checkpoint-2000	0.78	0.70	0.85	...
...
```

Where:
- The first column is named "CHECKPOINT" and contains checkpoint identifiers
- Subsequent columns are named "TASK_1", "TASK_2", etc., and contain performance metrics for each task
- Values are tab-separated

## Example Files

- `perf1.txt`: Example performance data for model checkpoints across multiple tasks
- `perf2.txt`: Another example dataset with different performance characteristics

You can add your own data files to this directory and use them with the examples by modifying the filename in the example scripts.
