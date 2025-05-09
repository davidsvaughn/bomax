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

def count_line_curve_intersections(x_values, y_values, num_trials=100):
    """
    Estimate the complexity of a curve by:
    1. Randomly selecting two points on the curve
    2. Drawing a straight line between them
    3. Counting intersections between this line and the curve
    4. Repeating and tracking the maximum count
    
    Args:
        x_values: x-coordinates of the curve points
        y_values: y-coordinates of the curve points
        num_trials: number of random line trials
    
    Returns:
        max_intersections: maximum number of intersections found
    """
    max_intersections = 0
    
    # Convert to numpy arrays if they aren't already
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    n_points = len(x_values)
    
    for _ in range(num_trials):
        # Pick two random indices (ensuring they're different)
        idx1, idx2 = random.sample(range(n_points), 2)
        
        # Get the two points on the curve
        x1, y1 = x_values[idx1], y_values[idx1]
        x2, y2 = x_values[idx2], y_values[idx2]
        
        # Skip if the x-values are the same (vertical line case)
        if x1 == x2:
            continue
        
        # Create a linear function between these two points
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Generate y-values for the straight line at each x-value
        line_y = slope * x_values + intercept
        
        # Calculate difference between curve and line
        diff = y_values - line_y
        
        # Count sign changes (intersections)
        # We only count where diff is not zero to avoid counting tangent points
        non_zero_diff = diff[diff != 0]
        if len(non_zero_diff) > 0:
            intersections = np.sum(np.diff(np.signbit(non_zero_diff)) != 0)
            
            # We need to add 1 to account for the first intersection
            # (since diff() reduces the length by 1)
            intersections += 1
            
            # Account for the two points we selected (they're not true intersections)
            intersections = max(0, intersections - 1)
            
            max_intersections = max(max_intersections, intersections)
    
    return max_intersections

import numpy as np
import random
import torch

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
    intersections = 1 + flips.sum(axis=1)           # (T,)
    # subtract 1 to ignore the two anchor points
    intersections = np.maximum(0, intersections - 1)

    return int(intersections.max())

# degree max
def maximum_degree_old(model, X_inputs, 
                  m=None, 
                  num_trials=500,
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
    
    # 
    x = to_numpy(X_inputs[X_inputs[:,1]==0][:,0])

    # loop over tasks
    degrees = []
    for i in range(m):
        y = to_numpy(mean[:, i])
        d = max_intersections_np(x, y, num_trials=num_trials)
        degrees.append(d)
        max_degree = np.max(degrees)
        
    stats = adict({'max': max_degree})
    if ret is None: 
        return stats
    else:
        stats.copy_to(ret)
        return ret
    
def max_intersections_shared_gpu(x, Y,                 # (N,), (N, C)
                                 num_trials     = 200_000,
                                 curve_blk      = 256,
                                 device         = "cuda"):
    """
    Shared-line version: every curve is tested against the *same* set of random
    lines, defined once from a reference curve (here: the mean curve).

    Returns
    -------
    int  – global maximum #intersections **including** anchors
    """

    # ---- tensors on GPU ----------------------------------------------------
    x  = torch.as_tensor(x, dtype=torch.float16, device=device)          # (N,)
    Y  = torch.as_tensor(Y, dtype=torch.float16, device=device)          # (N,C)
    N, C = Y.shape
    mx = torch.zeros(C, dtype=torch.int32, device=device)

    # ---- 1. choose reference curve & pre-compute the lines -----------------
    ref = Y.mean(dim=1)                 # (N,)  ← could pick Y[:,0] or PCA1 etc.

    rng  = torch.Generator(device=device)
    idx1 = torch.randint(0, N, (num_trials,), generator=rng, device=device)
    idx2 = torch.randint(0, N, (num_trials,), generator=rng, device=device)
    dup  = idx1 == idx2
    while dup.any():                    # enforce distinct anchors
        idx2[dup] = torch.randint(0, N, (dup.sum(),), generator=rng, device=device)
        dup = idx1 == idx2

    dx = x[idx2] - x[idx1]
    keep = dx != 0                      # kill verticals once for all curves
    idx1, idx2, dx = idx1[keep], idx2[keep], dx[keep]
    # T = idx1.numel()

    # slopes & intercepts – *single* vector
    slope = (ref[idx2] - ref[idx1]) / dx          # (T,)
    b     = ref[idx1] - slope * x[idx1]           # (T,)

    # line_y cached once: shape (T, N)
    line_y = slope[:, None] * x[None, :] + b[:, None]

    # ---- 2. process curves in small blocks to bound memory -----------------
    for c0 in range(0, C, curve_blk):
        c1 = min(c0 + curve_blk, C)
        Y_blk = Y[:, c0:c1]                       # (N, curve_blk)

        # broadcasting: (T, N, 1) vs (1, N, B) → (T, N, B)
        diff   = Y_blk[None, :, :] - line_y[:, :, None]
        signs  = torch.sign(diff)
        flips  = (signs[:, :-1, :] * signs[:, 1:, :] < 0).sum(1)   # (T,B)
        inter  = flips                                            # anchors kept
        mx[c0:c1] = torch.maximum(mx[c0:c1], inter.max(0).values)

    return mx.max().item()      # or return mx.cpu() for per-curve list

def max_intersections_shared_np(
        x, Y,
        num_trials    = 200_000,
        trial_block   = 20_000,   # how many random lines to process at once
        curve_block   = 256,       # how many curves (=columns) per sub-batch
        keep_anchors  = True,     # False → subtract 1 like the earlier code
        rng           = None,
        dtype         = np.float32):
    """
    Shared-line intersection estimator (CPU / NumPy).

    x : (N,) array-like               – common x-grid
    Y : (N, C) array-like             – each column is one curve
    Returns
    -------
    int  – global maximum intersection count across all curves.
          (Change the final line if you need the per-curve vector.)
    """

    # -------------- input preparation ------------------------------------
    x  = np.asarray(x, dtype=dtype)
    Y  = np.asarray(Y, dtype=dtype)
    N, C = Y.shape
    if rng is None:
        rng = np.random.default_rng()

    # use the mean curve as reference (any deterministic choice is fine)
    ref = Y.mean(axis=1)                         # (N,)

    # -------------- generate one common set of random lines --------------
    idx1 = rng.integers(0, N, size=num_trials, dtype=np.int64)
    idx2 = rng.integers(0, N, size=num_trials, dtype=np.int64)
    dup  = idx1 == idx2
    while dup.any():                             # ensure idx1 ≠ idx2
        idx2[dup] = rng.integers(0, N, size=dup.sum(), dtype=np.int64)
        dup = idx1 == idx2

    dx      = x[idx2] - x[idx1]
    valid   = dx != 0                            # kill verticals once for all
    idx1, idx2, dx = idx1[valid], idx2[valid], dx[valid]
    T       = idx1.size                          # actual trial count

    slope   = (ref[idx2] - ref[idx1]) / dx       # (T,)
    intercept = ref[idx1] - slope * x[idx1]      # (T,)
    # pre-compute line_y for trial blocks
    line_y_full = slope[:, None] * x[None, :] + intercept[:, None]   # (T, N)

    # -------------- main two-level batching loop -------------------------
    max_per_curve = np.zeros(C, dtype=np.int32)

    for t0 in range(0, T, trial_block):
        t1 = min(t0 + trial_block, T)
        line_y = line_y_full[t0:t1]              # (T_block, N)

        for c0 in range(0, C, curve_block):
            c1      = min(c0 + curve_block, C)
            Y_blk   = Y[:, c0:c1]                # (N, B)  view

            # broadcast: (T_block, N, 1) minus (1, N, B) → (T_block, N, B)
            diff    = Y_blk[None, :, :] - line_y[:, :, None]
            signs   = np.sign(diff)
            flips   = (signs[:, :-1, :] * signs[:, 1:, :] < 0).sum(axis=1)  # (T_block, B)

            inter   = flips if keep_anchors else np.maximum(0, flips - 1)
            max_per_curve[c0:c1] = np.maximum(max_per_curve[c0:c1],
                                               inter.max(axis=0))

    return int(max_per_curve.max())      # replace with max_per_curve if needed

def curve_metric(x, y, eps=1e-12, **kwargs):
    """
    Fast (O[n]) upper–bound for the maximum #intersections between an
    arbitrary straight line and the curve (x, y).

    Matches `max_intersections_np` for the usual cases:
      • convex / concave monotone curves  (→ 2)
      • S-shaped monotone curves          (→ 2 + #inflections)
      • Wiggly curves with several extrema (→ #monotone-segments)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---- 1. first-derivative sign pattern --------------------------------
    slopes = np.diff(y) / np.diff(x)
    s1 = np.sign(slopes)
    s1[s1 == 0] = 1                             # treat flats as tiny +ve slopes

    mono_flips = np.count_nonzero(s1[:-1] * s1[1:] < 0)
    monotone_segs = mono_flips + 1              # pieces of monotonicity

    # ---- 2. if curve already has >1 monotone piece, that is the answer ---
    if monotone_segs > 1:
        return monotone_segs                    # matches EI curves, sin-waves…

    # ---- 3. strictly monotone → look at inflections ----------------------
    # 2nd-derivative (vectorised three-point formula, O[n])
    curvature = (y[:-2] - 2*y[1:-1] + y[2:]) / (
                  (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) + eps)

    s2 = np.sign(curvature)
    s2[s2 == 0] = 1
    inflips = np.count_nonzero(s2[:-1] * s2[1:] < 0)

    # strictly-monotone convex/concave  → 2
    # strictly-monotone S-shape        → 2 + (#inflections)
    return 2 + inflips


def maximum_degree(model, X_inputs, 
                   m=None, 
                   num_trials=100,
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

    # loop over tasks
    x = to_numpy(X_inputs[X_inputs[:,1]==0][:,0])
    # max_degree = max_intersections_shared_np(x, mean, num_trials=num_trials)
    
    # loop over tasks
    degrees = []
    for i in range(m):
        y = to_numpy(mean[:, i])
        # d = max_intersections_np(x, y, num_trials=num_trials)
        d = curve_metric(x, y)
        degrees.append(d)
        max_degree = np.max(degrees)
        
    stats = adict({'max': max_degree})
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
    
    score1, score2, score3 = 1,2,3
    start_time = time.time()
    score1 = count_line_curve_intersections(x, y1, num_trials=num_trials)
    score2 = count_line_curve_intersections(x, y2, num_trials=num_trials)
    score3 = count_line_curve_intersections(x, y3, num_trials=num_trials)
    end_time = time.time()
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    print(f"Time taken for non-vectorized version: {end_time - start_time:.4f} seconds")

    start_time = time.time()    
    score1 = count_line_curve_intersections_vectorized(x, y1, num_trials=num_trials)
    score2 = count_line_curve_intersections_vectorized(x, y2, num_trials=num_trials)
    score3 = count_line_curve_intersections_vectorized(x, y3, num_trials=num_trials)
    end_time = time.time()
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    print(f"Time taken for vectorized version: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()    
    score1 = count_line_curve_intersections_vec(x, y1, num_trials=num_trials)
    score2 = count_line_curve_intersections_vec(x, y2, num_trials=num_trials)
    score3 = count_line_curve_intersections_vec(x, y3, num_trials=num_trials)
    end_time = time.time()
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    print(f"Time taken for vec version: {end_time - start_time:.4f} seconds")
    
    # Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=f"Low wiggliness (score={score1})")
    plt.plot(x, y2, label=f"Medium wiggliness (score={score2})")
    plt.plot(x, y3, label=f"High wiggliness (score={score3})")
    plt.legend()
    plt.title("Curves with Different Wiggliness Levels")
    plt.show()

if __name__ == "__main__":
    # Demonstration
    demonstrate_with_example()