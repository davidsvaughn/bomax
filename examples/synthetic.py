import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def make_base_curve(t, rng):
    """
    Draws a single ‘learning-curve-shaped’ function that:
        – rises, peaks, then tails off
    Uses a simple (t^α)*exp(-βt) parametric form.
    """
    # α = rng.uniform(1.5, 4.5)          # controls how quickly it rises
    # β = rng.uniform(3.0, 8.0)          # controls how quickly it falls
    α = rng.uniform(1, 4)          # controls how quickly it rises
    β = rng.uniform(4, 8)          # controls how quickly it falls
    base = (t ** α) * np.exp(-β * t)
    base /= base.max()                 # normalise to [0,1]
    shift = rng.uniform(0.4, 0.8)    # shift entire curve up
    scale = rng.uniform(0.2, 0.4)    # scale the hump height
    return shift + scale * base        # final prototype

def generate_learning_curves(n_steps=50,
                             n_curves=20, *,
                             avg_cluster_size=3,
                             cluster_size_var=1.0,
                             noise_level=0.1,
                             noise_smoothing=1.0,
                             jitter=0.05,
                             return_clusters=False,
                             seed=None):
    """
    Synthetic, correlated learning curves.

    Parameters
    ----------
    n_steps         int
        Number of time-steps (x-axis) per curve.
    n_curves         int
        Total number of curves to generate.
    avg_cluster_size float
        Target average number of curves per cluster.
    cluster_size_var float
        Variance in cluster sizes (higher = more variation).
    noise_level      float
        Std-dev of raw white noise added to each curve.
    noise_smoothing  float
        σ for Gaussian smoothing of the noise (higher ⇒ smoother).
    seed             int | None
        RNG seed for reproducibility.

    Returns
    -------
    t        (n_steps,)          – common x-axis (0 … 1)
    curves   (n_curves, n_steps)
    group_id (n_curves,)         – integer label per curve
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_steps)          # "training steps"

    # Determine number of clusters and their sizes
    n_clusters = max(1, round(n_curves / avg_cluster_size))
    
    # Generate random cluster sizes that sum to n_curves
    cluster_sizes = []
    remaining = n_curves
    
    for i in range(n_clusters - 1):
        # Generate a size with some randomness around the average
        size = max(1, round(avg_cluster_size + rng.normal(0, cluster_size_var)))
        # Make sure we don't exceed remaining curves
        size = min(size, remaining - (n_clusters - i - 1))
        cluster_sizes.append(size)
        remaining -= size
    
    # Add the last cluster with remaining curves
    cluster_sizes.append(remaining)
    
    # Shuffle the cluster sizes for more randomness
    rng.shuffle(cluster_sizes)
    
    curves   = []
    group_id = []

    for g, n_curves_in_cluster in enumerate(cluster_sizes):
        prototype = make_base_curve(t, rng)     # cluster latent
        for _ in range(n_curves_in_cluster):
            noise = rng.normal(0.0, noise_level, size=n_steps)
            noise = gaussian_filter1d(noise, noise_smoothing)
            # add uniform jitter to each point
            noise += rng.uniform(-jitter, jitter, size=n_steps)
            curve = prototype + noise
            curve -= max(0, curve.max() - 1)
            curves.append(curve)
            group_id.append(g)

    curves = np.vstack(curves)
    x_steps = np.arange(curves.shape[1])
    if return_clusters:
        return x_steps, curves.T, np.asarray(group_id)
    else:
        return x_steps, curves.T


# ---------------------------------------------------------------------
# Quick demonstration – generate and plot synthetic curves
# ---------------------------------------------------------------------
if __name__ == "__main__":
    t, curves, ids = generate_learning_curves(return_clusters=True, noise_smoothing=1, noise_level=0.1)

    for g in np.unique(ids):
        mask = ids == g
        plt.plot(t, curves.T[mask].T)        # one colour per group

    plt.plot(t, curves.T.mean(axis=0), "--", linewidth=2.5, color="red")
    plt.xlabel("training step")
    plt.ylabel("validation metric")
    plt.title("Synthetic correlated learning curves")
    plt.tight_layout()
    plt.show()
