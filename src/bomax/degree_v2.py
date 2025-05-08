import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
import time
import gpytorch
try:
    from .utils import adict, to_numpy
except ImportError:
    # For when the file is run directly as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bomax.utils import adict, to_numpy

import numpy as np
import random
import torch

import cupy as cp
import numba as nb

#==================================================================================================

def max_intersections_np(x, y, num_trials=10_000, rng=None, eps=1e-12):
    """
    Same goal as your original function but 100 % vectorised:

        • no Python loop over trials
        • sign-change counting done with boolean masks
        • zeros handled robustly with a tiny ε-shift

    Returns
    -------
    int  – maximum #intersections observed over all random lines
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size

    # -- 1. choose random *distinct* indices in vectorised form ------------
    idx1 = rng.integers(0, n, size=num_trials)
    # guarantee idx2 ≠ idx1 by resampling the collisions
    idx2 = rng.integers(0, n, size=num_trials)
    coll = idx2 == idx1
    while np.any(coll):
        idx2[coll] = rng.integers(0, n, size=coll.sum())
        coll = idx2 == idx1

    x1, y1, x2, y2 = x[idx1], y[idx1], x[idx2], y[idx2]
    dx = x2 - x1

    # -- 2. slopes/intercepts (vertical handled via +Inf) -------------------
    slope = np.divide(y2 - y1, dx, out=np.full_like(dx, np.inf), where=dx!=0)
    intercept = y1 - slope * x1      # if slope==inf the values won't be used

    # -- 3. evaluate curve–line differences for *all* trials ----------------
    # broadcast: (T,1)·(1,N) → (T,N)
    line_y = slope[:, None] * x[None, :] + intercept[:, None]
    diff   = y[None, :] - line_y              # shape (T,N)

    # -- 4. count sign changes in one shot ----------------------------------
    # treat exact zeros by nudging them to ±eps with the *previous* sign
    zmask = diff == 0
    if zmask.any():
        # copy to avoid modifying original
        diff = diff.copy()
        # propagate previous non-zero sign forward
        signs = np.sign(diff)
        signs[:, 0] = np.where(signs[:, 0]==0, 1, signs[:, 0])
        signs[:, 1:] = np.where(signs[:, 1:]==0, signs[:, :-1], signs[:, 1:])
        diff[zmask] = eps * signs[zmask]

    signs = np.sign(diff)
    # Boolean matrix True where sign change between consecutive points
    flips = signs[:, :-1] * signs[:, 1:] < 0
    intersections = flips.sum(axis=1)           # (T,)
    # subtract 1 to ignore the two anchor points
    intersections = np.maximum(0, intersections - 1)

    return int(intersections.max())

#==================================================================================================

def max_intersections_cp(x, y, num_trials=10_000, rng=None, eps=1e-12):
    """
    Same goal as your original function but 100 % vectorised:

        • no Python loop over trials
        • sign-change counting done with boolean masks
        • zeros handled robustly with a tiny ε-shift

    Returns
    -------
    int  – maximum #intersections observed over all random lines
    """
    if rng is None:
        rng = cp.random.default_rng()

    x = cp.asarray(x, dtype=cp.float64)
    y = cp.asarray(y, dtype=cp.float64)
    n = x.size

    # -- 1. choose random *distinct* indices in vectorised form ------------
    idx1 = rng.integers(0, n, size=num_trials)
    # guarantee idx2 ≠ idx1 by resampling the collisions
    idx2 = rng.integers(0, n, size=num_trials)
    coll = idx2 == idx1
    while cp.any(coll):
        idx2[coll] = rng.integers(0, n, size=coll.sum())
        coll = idx2 == idx1

    x1, y1, x2, y2 = x[idx1], y[idx1], x[idx2], y[idx2]
    dx = x2 - x1

    # -- 2. slopes/intercepts (vertical handled via +Inf) -------------------
    slope = cp.divide(y2 - y1, dx, out=cp.full_like(dx, cp.inf), where=dx!=0)
    intercept = y1 - slope * x1      # if slope==inf the values won't be used

    # -- 3. evaluate curve–line differences for *all* trials ----------------
    # broadcast: (T,1)·(1,N) → (T,N)
    line_y = slope[:, None] * x[None, :] + intercept[:, None]
    diff   = y[None, :] - line_y              # shape (T,N)

    # -- 4. count sign changes in one shot ----------------------------------
    # treat exact zeros by nudging them to ±eps with the *previous* sign
    zmask = diff == 0
    if zmask.any():
        # copy to avoid modifying original
        diff = diff.copy()
        # propagate previous non-zero sign forward
        signs = cp.sign(diff)
        signs[:, 0] = cp.where(signs[:, 0]==0, 1, signs[:, 0])
        signs[:, 1:] = cp.where(signs[:, 1:]==0, signs[:, :-1], signs[:, 1:])
        diff[zmask] = eps * signs[zmask]

    signs = cp.sign(diff)
    # Boolean matrix True where sign change between consecutive points
    flips = signs[:, :-1] * signs[:, 1:] < 0
    intersections = flips.sum(axis=1)           # (T,)
    # subtract 1 to ignore the two anchor points
    intersections = cp.maximum(0, intersections - 1)

    return int(intersections.max())

#==================================================================================================

def max_intersections_torch(x, y, num_trials=10_000, device="cuda"):
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)

    n = x.numel()
    idx1 = torch.randint(0, n, (num_trials,), device=device)
    idx2 = torch.randint(0, n, (num_trials,), device=device)
    # idem: enforce distinctness (vectorised)

    x1, y1, x2, y2 = x[idx1], y[idx1], x[idx2], y[idx2]
    dx = x2 - x1
    slope = (y2 - y1) / dx
    slope[dx == 0] = torch.inf
    intercept = y1 - slope * x1

    diff = y - (slope[:, None] * x + intercept[:, None])
    signs = torch.sign(diff)
    flips = signs[:, :-1] * signs[:, 1:] < 0
    intersections = flips.sum(1).clamp(min=0) - 1
    return int(intersections.max().item())

#==================================================================================================

# ---------------------------------------------------------------------
# 1.  JIT-compiled kernel – *no* Python objects allowed inside here
# ---------------------------------------------------------------------
@nb.njit(parallel=True, fastmath=True)
def _max_intersections_numba(x, y, idx1, idx2):
    """
    Parameters
    ----------
    x, y : 1-D float64 arrays of equal length (the curve)
    idx1, idx2 : 1-D int64 arrays (same length = #trials)
                 each (idx1[t], idx2[t]) is a distinct anchor pair

    Returns
    -------
    int – maximum intersections over all trials
    """
    T = idx1.size          # num_trials
    N = x.size             # curve length
    best = 0

    for t in nb.prange(T):  # parallel over trials
        i = idx1[t]
        j = idx2[t]

        # --- slope / intercept (vertical line → skip) -------------------
        dx = x[j] - x[i]
        if dx == 0.0:          # vertical: no intersections counted
            continue

        m = (y[j] - y[i]) / dx
        b = y[i] - m * x[i]

        # --- count sign flips along the curve --------------------------
        prev_sign = 0
        flips = 0

        for k in range(N):
            diff = y[k] - (m * x[k] + b)

            # skip exact zeros (anchor points or genuine hits)
            if diff == 0.0:
                continue

            s = 1 if diff > 0.0 else -1
            if prev_sign != 0 and s != prev_sign:
                flips += 1
            prev_sign = s

        # subtract one to ignore the two anchor points
        inter = flips - 1 if flips > 0 else 0
        if inter > best:
            best = inter

    return best


# ---------------------------------------------------------------------
# 2.  Public convenience wrapper
# ---------------------------------------------------------------------
def max_intersections_numba(x, y, num_trials=10_000, rng=None):
    """
    JIT-accelerated version that keeps memory usage O(N_points)
    and avoids the huge (trials × N) matrix.

    Example
    -------
    >>> max_line_intersections_numba(xs, ys, 50_000)
    7
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3:
        raise ValueError("Need at least three curve points")

    # generate random, *distinct* anchor indices
    idx1 = rng.integers(0, n, size=num_trials, dtype=np.int64)
    idx2 = rng.integers(0, n, size=num_trials, dtype=np.int64)
    coll = idx1 == idx2
    while np.any(coll):
        idx2[coll] = rng.integers(0, n, size=coll.sum(), dtype=np.int64)
        coll = idx1 == idx2

    # hand everything to the compiled kernel
    return int(_max_intersections_numba(x, y, idx1, idx2))

#==================================================================================================

def count_line_curve_intersections_vectorized(x_values, y_values, num_trials=500):
    """
    Estimate the complexity of a curve by:
    1. Randomly selecting two points on the curve
    2. Drawing a straight line between them
    3. Counting intersections between this line and the curve
    4. Repeating and tracking the maximum count

    PARTIALLY VECTORIZED VERSION:
      - We pick all random pairs of points at once
      - Compute slopes, intercepts, line values in NumPy arrays
      - Then do a small Python loop (length = num_trials) to count sign changes
        after filtering out zeros for each line separately.

    Args:
        x_values: x-coordinates of the curve points (array-like)
        y_values: y-coordinates of the curve points (array-like)
        num_trials: number of random line trials

    Returns:
        max_intersections: maximum number of intersections found
    """
    # Convert inputs to NumPy arrays
    # if CUDA arrays, convert to cpu...
    if isinstance(x_values, torch.Tensor):
        x_values = x_values.cpu().numpy()
    if isinstance(y_values, torch.Tensor):
        y_values = y_values.cpu().numpy()
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    n_points = len(x_values)

    # 1) Generate random pairs of *distinct* indices for all trials at once
    #    (Here we just do it with Python's 'random.sample' in a loop; you could
    #     also use np.random.choice, etc. But let's keep it simple.)
    all_pairs = []
    for _ in range(num_trials):
        i1, i2 = random.sample(range(n_points), 2)
        all_pairs.append((i1, i2))
    all_pairs = np.array(all_pairs, dtype=int)  # shape (num_trials, 2)
    
    # Extract x1,y1 and x2,y2 for each trial in vectorized form
    idx1 = all_pairs[:, 0]
    idx2 = all_pairs[:, 1]
    x1 = x_values[idx1]  # shape (num_trials,)
    y1 = y_values[idx1]
    x2 = x_values[idx2]
    y2 = y_values[idx2]

    # 2) Compute slopes & intercepts in a vectorized way
    #    We'll need to handle the vertical-line case (x2 == x1) by masking them out.
    dx = x2 - x1
    vertical_mask = (dx == 0)
    
    # Avoid division by zero: for vertical lines, we won't try to compute slope
    # (We'll skip them from intersection counting, as the original code did.)
    slope = np.empty(num_trials, dtype=float)
    slope[vertical_mask] = np.nan  # placeholder
    slope[~vertical_mask] = (y2[~vertical_mask] - y1[~vertical_mask]) / dx[~vertical_mask]

    intercept = np.empty(num_trials, dtype=float)
    intercept[vertical_mask] = np.nan
    intercept[~vertical_mask] = y1[~vertical_mask] - slope[~vertical_mask] * x1[~vertical_mask]
    
    # 3) For each line, compute "diff = curve_y - line_y" for all curve points
    #    Doing so in a fully vectorized manner => an array of shape (num_trials, n_points).
    #    line_y[i, :] = slope[i] * x_values + intercept[i].
    
    # Expand slopes and intercepts along a new axis, so we can broadcast against x_values:
    # slope[i, None] times x_values[None, :] => shape (num_trials, n_points)
    # intercept[i, None] => shape (num_trials, 1)
    line_y = slope[:, None] * x_values[None, :] + intercept[:, None]  # shape (num_trials, n_points)
    
    # diff[i, j] = y_values[j] - line_y[i, j]
    # => shape (num_trials, n_points)
    # We want y_values to broadcast across the second dimension (points),
    # so let's do y_values[None, :] minus line_y.
    diff = y_values[None, :] - line_y
    
    # 4) Count the sign changes in each row of `diff`, after skipping zeros, in a small loop
    max_intersections = 0
    
    for i in range(num_trials):
        # If it's a vertical line (or we assigned slope=nan), skip
        if vertical_mask[i]:
            continue
        
        # Filter out zeros:
        non_zero_diff = diff[i][diff[i] != 0]
        if len(non_zero_diff) == 0:
            continue
        
        # Count sign changes
        signs = np.sign(non_zero_diff)  # +1 or -1
        changes = np.diff(signs)        # difference between consecutive signs
        # Each place changes != 0 indicates a sign change
        num_sign_changes = np.count_nonzero(changes != 0)
        
        # Add 1 for the "number of sign segments" => # of times crossing zero
        intersections = num_sign_changes + 1
        
        # Subtract 1 to avoid counting the endpoints we used to define the line
        # (Matching your original logic.  The rationale: we don't want to
        #  treat the chosen endpoints as if they 'intersected' the curve.)
        intersections = max(0, intersections - 1)
        
        max_intersections = max(max_intersections, intersections)
    
    return max_intersections

def fill_zeros(arr):
    """
    Fill zero values in a 2D NumPy array according to specific rules:
    1. If a zero is in the rightmost column, fill with first nonzero value to the left (same row)
    2. For all other zeros, fill with the first nonzero value to the right (same row)
    
    Parameters:
    -----------
    arr : numpy.ndarray
        2D array with some zero values
        
    Returns:
    --------
    numpy.ndarray
        Array with zeros filled according to the rules
    """
    # Make a copy to avoid modifying the original array
    result = arr.copy()
    rows, cols = arr.shape
    
    # Process each row separately for vectorization
    for i in range(rows):
        row = result[i, :]
        
        # Handle zeros in positions other than the rightmost column (rule 2)
        # For each position, find the first non-zero value to the right
        for j in range(cols - 1):
            if row[j] == 0:
                # Find first non-zero value to the right
                nonzeros_to_right = row[j+1:][row[j+1:] != 0]
                if len(nonzeros_to_right) > 0:
                    row[j] = nonzeros_to_right[0]
        
        # Handle zero in the rightmost column (rule 1)
        if row[-1] == 0:
            # Find first non-zero value to the left
            nonzeros_to_left = row[:-1][row[:-1] != 0]
            if len(nonzeros_to_left) > 0:
                row[-1] = nonzeros_to_left[-1]  # Get the rightmost non-zero value
    
    return result


# More efficient, vectorized implementation
def fill_zeros_vectorized(arr):
    """
    Vectorized implementation to fill zero values in a 2D NumPy array:
    1. If a zero is in the rightmost column, fill with first nonzero value to the left (same row)
    2. For all other zeros, fill with the first nonzero value to the right (same row)
    
    This implementation is significantly faster for large arrays.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        2D array with some zero values
        
    Returns:
    --------
    numpy.ndarray
        Array with zeros filled according to the rules
    """
    result = arr.copy()
    rows, cols = arr.shape
    
    # Process each row
    for i in range(rows):
        row = result[i, :]
        zero_indices = np.where(row == 0)[0]
        
        if len(zero_indices) == 0:
            continue
            
        # Create a mask of non-zero values
        nonzero_mask = row != 0
        nonzero_indices = np.where(nonzero_mask)[0]
        
        if len(nonzero_indices) == 0:
            continue  # Skip if row is all zeros
            
        # Process rightmost column if it contains zero (rule 1)
        if cols-1 in zero_indices:
            # Find rightmost non-zero value to the left
            left_nonzeros = nonzero_indices[nonzero_indices < cols-1]
            if len(left_nonzeros) > 0:
                result[i, cols-1] = row[left_nonzeros[-1]]
                
            # Remove the rightmost column from zero_indices for the next step
            zero_indices = zero_indices[zero_indices != cols-1]
            
        # Process all other zeros (rule 2)
        for j in zero_indices:
            # Find indices of non-zeros to the right
            right_nonzeros = nonzero_indices[nonzero_indices > j]
            if len(right_nonzeros) > 0:
                result[i, j] = row[right_nonzeros[0]]  # First non-zero to the right
                
    return result

def count_line_curve_intersections_vec(x_values, y_values, num_trials=100):
    """
    Estimate the complexity of a curve by:
    1. Randomly selecting two points on the curve
    2. Drawing a straight line between them
    3. Counting intersections between this line and the curve
    4. Repeating and tracking the maximum count

    PARTIALLY VECTORIZED VERSION:
      - We pick all random pairs of points at once
      - Compute slopes, intercepts, line values in NumPy arrays
      - Then do a small Python loop (length = num_trials) to count sign changes
        after filtering out zeros for each line separately.

    Args:
        x_values: x-coordinates of the curve points (array-like)
        y_values: y-coordinates of the curve points (array-like)
        num_trials: number of random line trials

    Returns:
        max_intersections: maximum number of intersections found
    """
    # Convert inputs to NumPy arrays
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    n_points = len(x_values)

    # 1) Generate random pairs of *distinct* indices for all trials at once
    #    (Here we just do it with Python's 'random.sample' in a loop; you could
    #     also use np.random.choice, etc. But let's keep it simple.)
    all_pairs = []
    for _ in range(num_trials):
        i1, i2 = random.sample(range(n_points), 2)
        all_pairs.append((i1, i2))
    all_pairs = np.array(all_pairs, dtype=int)  # shape (num_trials, 2)
    
    # Extract x1,y1 and x2,y2 for each trial in vectorized form
    idx1 = all_pairs[:, 0]
    idx2 = all_pairs[:, 1]
    x1 = x_values[idx1]  # shape (num_trials,)
    y1 = y_values[idx1]
    x2 = x_values[idx2]
    y2 = y_values[idx2]

    # 2) Compute slopes & intercepts in a vectorized way
    #    We'll need to handle the vertical-line case (x2 == x1) by masking them out.
    dx = x2 - x1
    vertical_mask = (dx == 0)
    
    # Avoid division by zero: for vertical lines, we won't try to compute slope
    # (We'll skip them from intersection counting, as the original code did.)
    slope = np.empty(num_trials, dtype=float)
    slope[vertical_mask] = np.nan  # placeholder
    slope[~vertical_mask] = (y2[~vertical_mask] - y1[~vertical_mask]) / dx[~vertical_mask]

    intercept = np.empty(num_trials, dtype=float)
    intercept[vertical_mask] = np.nan
    intercept[~vertical_mask] = y1[~vertical_mask] - slope[~vertical_mask] * x1[~vertical_mask]
    
    # 3) For each line, compute "diff = curve_y - line_y" for all curve points
    #    Doing so in a fully vectorized manner => an array of shape (num_trials, n_points).
    #    line_y[i, :] = slope[i] * x_values + intercept[i].
    
    # Expand slopes and intercepts along a new axis, so we can broadcast against x_values:
    # slope[i, None] times x_values[None, :] => shape (num_trials, n_points)
    # intercept[i, None] => shape (num_trials, 1)
    line_y = slope[:, None] * x_values[None, :] + intercept[:, None]  # shape (num_trials, n_points)
    
    # diff[i, j] = y_values[j] - line_y[i, j]
    # => shape (num_trials, n_points)
    # We want y_values to broadcast across the second dimension (points),
    # so let's do y_values[None, :] minus line_y.
    diff = y_values[None, :] - line_y
    
    # 4) Count the sign changes in each row of `diff`, after skipping zeros, in a small loop
    diff = diff[vertical_mask == False]
    
    non_zero_diff = fill_zeros_vectorized(diff)
    signs = np.sign(non_zero_diff)  # +1 or -1
    changes = np.diff(signs, axis=1)        # difference between consecutive signs
    intersections = np.count_nonzero(changes != 0, axis=1)# + 1
    # intersections = np.maximum(0, np.count_nonzero(changes != 0, axis=1))
    max_intersections = np.max(intersections)
    return max_intersections

# degree metric
def degree_metric(model, X_inputs, 
                  m=None, 
                  num_trials=500,
                  mean_max=None,
                  max_max=None,
                  ret=None,
                  verbose=False):
    if m is None:
        m = int(X_inputs[:,1].max().item() + 1)
        
    # make posterior predictions
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(X_inputs))
    mean = pred.mean.reshape(-1, m)
    model.train()
    
    avg_degree, max_degree, mean_degree = -1,-1,-1
    
    # mean_degree = degree of mean regression line (mean of all curves across all tasks)
    x = to_numpy(X_inputs[X_inputs[:,1]==0][:,0])
    mean_degree = count_line_curve_intersections_vectorized(x, mean.mean(axis=1), num_trials=num_trials)

    # loop over tasks
    degrees = []
    for i in range(m):
        y = to_numpy(mean[:, i])
        d = count_line_curve_intersections_vectorized(x, y, num_trials=num_trials)
        degrees.append(d)
        max_degree = np.max(degrees)
            
    avg_degree = np.mean(degrees)
    
    # show histogram
    if verbose:
        print(f'Average degree: {avg_degree}')
        plt.hist(degrees, bins=np.ptp(degrees)+1)
        plt.show()
        
    stats = adict({'avg': avg_degree, 'max': max_degree, 'mean': mean_degree})
    if ret is None: 
        return stats
    else:
        stats.copy_to(ret)
        return ret

     
# Example usage:
def demonstrate_with_example():
    # Generate example data with increasing wiggliness
    x = np.linspace(0, 10, 1000)
    
    # Three curves with different levels of wiggliness
    y1 = np.sin(x)                    # Low wiggliness
    y2 = np.sin(x) + 0.5 * np.sin(3 * x)  # Medium wiggliness
    y3 = np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.sin(7 * x)  # High wiggliness
    
    # Compute wiggliness scores
    num_trials = 1000
    
    # Plot the curves
    score1, score2, score3 = 1,2,3
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=f"Low wiggliness (score={score1})")
    plt.plot(x, y2, label=f"Medium wiggliness (score={score2})")
    plt.plot(x, y3, label=f"High wiggliness (score={score3})")
    plt.legend()
    plt.title("Curves with Different Wiggliness Levels")
    plt.show()
    

    func_list = [
        count_line_curve_intersections_vectorized,
        count_line_curve_intersections_vec,
        max_intersections_np,
        max_intersections_cp,
        max_intersections_torch,
        max_intersections_numba,
    ]
    
    for func in func_list:
        try:
            start_time = time.time()
            score1 = func(x, y1, num_trials=num_trials)
            score2 = func(x, y2, num_trials=num_trials)
            score3 = func(x, y3, num_trials=num_trials)
            end_time = time.time()
            print(f"Wiggliness scores:")
            print(f"Curve 1 (low): {score1}")
            print(f"Curve 2 (medium): {score2}")
            print(f"Curve 3 (high): {score3}")
            print(f"Time taken for {func.__name__}: {end_time - start_time:.4f} seconds")
        except KeyboardInterrupt:
            print(f"Skipping {func.__name__} due to keyboard interrupt")
            continue

if __name__ == "__main__":
    # Demonstration
    demonstrate_with_example()