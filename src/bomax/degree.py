import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import gpytorch
try:
    from .utils import adict, to_numpy
except ImportError:
    # For when the file is run directly as a script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bomax.utils import adict, to_numpy
    
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

    start_time = time.time()    
    score1 = curve_metric(x, y1, num_trials=num_trials)
    score2 = curve_metric(x, y2, num_trials=num_trials)
    score3 = curve_metric(x, y3, num_trials=num_trials)
    end_time = time.time()
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    print(f"Time taken for curve_metric: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()    
    score1 = max_intersections_np(x, y1, num_trials=num_trials)
    score2 = max_intersections_np(x, y2, num_trials=num_trials)
    score3 = max_intersections_np(x, y3, num_trials=num_trials)
    end_time = time.time()
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    print(f"Time taken for max_intersections_np: {end_time - start_time:.4f} seconds")
    
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