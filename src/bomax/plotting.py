import sys, os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import logging
from datetime import datetime
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kernel_regression(y, x=None, bandwidth=1.0, kernel='epanechnikov'):
    """
    Perform non-parametric kernel regression on 1D data.
    
    Parameters:
    -----------
    x : array-like
        The independent variable values of the training data.
    y : array-like
        The dependent variable values of the training data.
    x_new : array-like
        The points at which to evaluate the regression function.
    bandwidth : float, default=1.0
        The bandwidth parameter controlling the smoothness of the regression.
    kernel : str, default='gaussian'
        The kernel function to use. Options: 'gaussian', 'epanechnikov', 'uniform'.
        
    Returns:
    --------
    y_pred : array-like
        The predicted values at x_new.
    """
    x = np.asarray(x) if x is not None else np.arange(len(y))
    y = np.asarray(y)
    x_new = x
    
    
    # Define kernel functions
    def gaussian_kernel(u):
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    
    def epanechnikov_kernel(u):
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    
    def uniform_kernel(u):
        return np.where(np.abs(u) <= 1, 0.5, 0)
    
    # Select the kernel function
    if kernel == 'gaussian':
        kernel_func = gaussian_kernel
    elif kernel == 'epanechnikov':
        kernel_func = epanechnikov_kernel
    elif kernel == 'uniform':
        kernel_func = uniform_kernel
    else:
        raise ValueError("Kernel must be 'gaussian', 'epanechnikov', or 'uniform'")
    
    # Perform kernel regression
    y_pred = np.zeros_like(x_new, dtype=float)
    
    for i, x_i in enumerate(x_new):
        # Calculate distances
        distances = (x_i - x) / bandwidth
        
        # Calculate weights
        weights = kernel_func(distances)
        
        # Normalize weights
        if np.sum(weights) != 0:
            weights = weights / np.sum(weights)
        
        # Calculate prediction
        y_pred[i] = np.sum(weights * y)
    
    return y_pred

def smooth_spline(values, smooth_factor=3):
    """
    Apply spline-based smoothing to a 1D array.
    
    Args:
        values (array-like): The input array to smooth
        smooth_factor (float): Controls the smoothness. Higher values = smoother curve.
                              This parameter is used differently than in gaussian_filter1d.
                              
    Returns:
        numpy.ndarray: The smoothed array
    """
    # Create x coordinates (evenly spaced indices)
    x = np.arange(len(values))
    
    # Create a univariate spline
    # The s parameter controls the smoothness - higher values = smoother curve
    # We scale the smooth_factor to make it roughly comparable to gaussian_filter1d's sigma
    s = len(values) * smooth_factor * 0.1
    
    # Fit the spline (handling potential errors)
    try:
        spline = UnivariateSpline(x, values, s=s)
        # Evaluate the spline at the original x coordinates
        smoothed_values = spline(x)
        return smoothed_values
    except Exception as e:
        print(f"Spline smoothing failed: {e}. Falling back to original values.")
        return values

# Set seaborn style for better visualization
sns.set_theme(style="whitegrid")




def plot_multiple_columns(test_indices=[1, 2, 3, 4, 17, 18, 22, 28, 37, 40, 66], filename='phi4-math-4claude.txt', 
                         smooth_factor=3, smooth_method='gaussian', show_original=False):
    """
    Create a single plot showing multiple TEST columns with different colors.
    Each column is represented by a smooth line (no dots) with a different color.
    The columns are normalized to range between 0.5 and 0.8, and smoothed for better visualization.
    
    Args:
        test_indices (list or int): Either a list of TEST column indices to plot (e.g., [1, 2, 3] for TEST_1, TEST_2, TEST_3)
                                   or an integer N to randomly select N columns
        filename (str): Name of the data file to use
        smooth_factor (float): Smoothing factor (sigma for Gaussian, smoothness for spline)
        smooth_method (str): Smoothing method to use ('gaussian' or 'spline')
        show_original (bool): If True, also plots the original unsmoothed values as dashed lines
        
    Returns:
        None: Saves the plot to the plots directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    rand_seed = random.randint(1000, 10000) if rand_seed <= 0 else rand_seed
    rand_seed = 1521
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    print(f"Random seed: {rand_seed}")

    # Load data
    file_path = os.path.join(data_dir, filename)
    
    # Read the first line to get column names
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    # Fix the header by adding a tab between CHECKPOINT and TEST_AVERAGE if needed
    if 'CHECKPOINTTEST_AVERAGE' in header_line:
        header_line = header_line.replace('CHECKPOINTTEST_AVERAGE', 'CHECKPOINT\tTEST_AVERAGE')
        
    # Split the header by tabs to get column names
    column_names = header_line.split('\t')
    
    # Read the data with the corrected column names
    df = pd.read_csv(file_path, delimiter='\t', names=column_names, skiprows=1)
    
    # Extract checkpoint numbers for x-axis
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Get all column names and identify TEST columns
    all_columns = df.columns.tolist()
    all_test_columns = [col for col in all_columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    # Handle case where test_indices is an integer (randomly select N columns)
    if isinstance(test_indices, int):
        n_columns = min(test_indices, len(all_test_columns))  # Ensure we don't try to select more columns than available
        # Set a seed for reproducibility, but allow different runs to get different random selections
        selected_columns = random.sample(all_test_columns, n_columns)
        print(f"Randomly selected {n_columns} columns: {', '.join(selected_columns)}")
        
        # Collect data for the randomly selected columns
        data_to_plot = {}
        plotted_columns = []
        for col_name in selected_columns:
            data_to_plot[col_name] = df[col_name].values
            plotted_columns.append(col_name)
    else:
        # Handle case where test_indices is a list (use specified indices)
        data_to_plot = {}
        plotted_columns = []
        for test_idx in test_indices:
            col_name = f'TEST_{test_idx}'
            if col_name in df.columns:
                data_to_plot[col_name] = df[col_name].values
                plotted_columns.append(col_name)
            else:
                print(f"Warning: Column {col_name} not found in the data file.")
    
    # Normalize all columns to range between 0.5 and 0.8, with jitter for better visibility
    normalized_data = {}
    original_data = {}  # Store original data for optional plotting
    # Set a random seed based on column names for consistent jitter across runs
    # np.random.seed(42)
    
    for col_name, values in data_to_plot.items():
        # Generate a unique seed for each column based on its name
        col_seed = sum(ord(c) for c in col_name)
        np.random.seed(col_seed)
        
        # Create jittered min and max values for normalization
        # This gives each curve a slightly different range for better visibility
        min_jitter = np.random.uniform(-0.05, 0.05)  # Small jitter for min value
        max_jitter = np.random.uniform(-0.05, 0.05)  # Small jitter for max value
        
        # Ensure min_value is always less than max_value by at least 0.15
        min_value = 0.5 + min_jitter
        max_value = 0.8 + max_jitter
        
        # Adjust if the range gets too small
        if max_value - min_value < 0.15:
            max_value = min_value + 0.15
        
        # Apply smoothing based on selected method
        if smooth_method.lower() == 'gaussian':
            smoothed_values = gaussian_filter1d(values, sigma=smooth_factor)
        elif smooth_method.lower() == 'spline':
            smoothed_values = smooth_spline(values, smooth_factor=smooth_factor)
        elif smooth_method.lower() == 'kernel':
            smoothed_values = kernel_regression(values, bandwidth=smooth_factor, kernel='epanechnikov')
        else:
            print(f"Warning: Unknown smoothing method '{smooth_method}'. Using Gaussian smoothing.")
            smoothed_values = gaussian_filter1d(values, sigma=smooth_factor)
        
        # Normalize both smoothed and original values to the jittered range
        # For smoothed values
        if np.max(smoothed_values) != np.min(smoothed_values):  # Avoid division by zero
            normalized_values = min_value + (max_value - min_value) * (smoothed_values - np.min(smoothed_values)) / (np.max(smoothed_values) - np.min(smoothed_values))
        else:
            normalized_values = np.full_like(smoothed_values, (min_value + max_value) / 2)  # Default to middle of range if all values are the same
        
        normalized_data[col_name] = normalized_values
        
        # For original values (if show_original is True)
        if show_original:
            if np.max(values) != np.min(values):  # Avoid division by zero
                # Use the same normalization range as the smoothed values for consistency
                original_normalized = min_value + (max_value - min_value) * (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                original_normalized = np.full_like(values, (min_value + max_value) / 2)
                
            original_data[col_name] = original_normalized
    
    # Reset random seed
    # np.random.seed(None)
    
    # Create a custom color palette with more distinct colors
    palette = sns.color_palette("husl", 5* len(normalized_data))
    # rotate the palette to avoid reds
    n = 3
    palette = palette[n:] + palette[:n]
    
    # Plot each normalized and smoothed column with a different color
    for i, (col_name, values) in enumerate(normalized_data.items()):
        # Plot smoothed values with solid lines
        plt.plot(X_feats, values, linewidth=2.5, label=col_name, color=palette[i])
        
        # If show_original is True, also plot original values with dashed lines of the same color
        if show_original and col_name in original_data:
            plt.plot(X_feats, original_data[col_name], linestyle='--', linewidth=1.5, color=palette[i], alpha=0.7)
    
    # Set plot properties with seaborn styling
    plt.xlabel('Model Checkpoint (Global Step)', fontsize=18)
    plt.ylabel('Performance', fontsize=18)
    plt.title(f'Benchmark Learning Curve', fontsize=20)
    # plt.legend(loc='best', frameon=True, framealpha=0.7)
    
    # Calculate the actual min and max values across all normalized data after jitter
    all_min = min(np.min(values) for values in normalized_data.values())
    all_max = max(np.max(values) for values in normalized_data.values())
    
    # Add a small margin to the limits
    margin = 0.02
    plt.ylim(all_min - margin, all_max + margin)
    
    # Add tight layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'multiple_columns_plot_{rand_seed}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved: {plot_path}")



def create_test_plots():
    """
    Create plots showing checkpoints on the x-axis and TEST column values on the y-axis.
    One plot is generated for each TEST column.
    All plots are saved in the multitask/plots directory.
    """
    fn = 'phi4-math-4claude.txt'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # load data
    df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')

    # Extract checkpoint numbers
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
     
    # Identify test columns (excluding average)
    test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    Y_test = df[test_cols].values
    n, m = Y_test.shape
    
    # Create a plot for each TEST column
    for i, col_name in enumerate(test_cols):
        plt.figure(figsize=(10, 6))
        plt.plot(X_feats, Y_test[:, i], marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Checkpoint', fontsize=12)
        plt.ylabel(f'{col_name} Score', fontsize=12)
        plt.title(f'Performance of {col_name} Across Checkpoints', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f'{col_name}_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot saved: {plot_path}")

def plot_clustered_columns(filename='phi4-math-4claude.txt', 
                          bandwidth=15, 
                          n_clusters=4, 
                          min_cluster_size=3,
                          max_cluster_size=6,
                          corr_min = 0.95,
                          selected_clusters=None,
                          show_legend=False,
                          ):
    """
    Create a plot showing clustered TEST columns based on correlation between
    Epanechnikov kernel regression smoothed curves, using K-means clustering.
    
    Args:
        filename (str): Name of the data file to use
        bandwidth (float): Bandwidth parameter for Epanechnikov kernel regression
        n_clusters (int): Number of clusters for K-means
        min_cluster_size (int): Minimum number of members per cluster to display
        max_cluster_size (int): Maximum number of members per cluster to display
        corr_min (float): Minimum correlation threshold for cluster members
        selected_clusters (list, optional): List of specific cluster indices to display.
                                           If None, all valid clusters are displayed.
        
    Returns:
        None: Saves the plot to the plots directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    
    rand_seed = random.randint(1000, 10000)
    rand_seed = 6197
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    print(f"Random seed: {rand_seed}")
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    file_path = os.path.join(data_dir, filename)
    
    # Read the first line to get column names
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    # Fix the header by adding a tab between CHECKPOINT and TEST_AVERAGE if needed
    if 'CHECKPOINTTEST_AVERAGE' in header_line:
        header_line = header_line.replace('CHECKPOINTTEST_AVERAGE', 'CHECKPOINT\tTEST_AVERAGE')
        
    # Split the header by tabs to get column names
    column_names = header_line.split('\t')
    
    # Read the data with the corrected column names
    df = pd.read_csv(file_path, delimiter='\t', names=column_names, skiprows=1)
    
    # Extract checkpoint numbers for x-axis
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Get all TEST columns (excluding average)
    all_columns = df.columns.tolist()
    all_test_columns = [col for col in all_columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    print(f"Processing {len(all_test_columns)} TEST columns...")
    
    # Apply Epanechnikov kernel regression to ALL columns
    smoothed_data = {}
    normalized_data = {}
    
    # Set a fixed range for normalization (no jitter needed for clustering)
    min_value = 0.5
    max_value = 0.8
    
    for col_name in all_test_columns:
        values = df[col_name].values
        
        # Apply Epanechnikov kernel regression
        smoothed_values = kernel_regression(values, bandwidth=bandwidth, kernel='epanechnikov')
        
        # Normalize to range [0.5, 0.8]
        if np.max(smoothed_values) != np.min(smoothed_values):  # Avoid division by zero
            normalized_values = min_value + (max_value - min_value) * (smoothed_values - np.min(smoothed_values)) / (np.max(smoothed_values) - np.min(smoothed_values))
        else:
            normalized_values = np.full_like(smoothed_values, (min_value + max_value) / 2)
            
        smoothed_data[col_name] = smoothed_values
        normalized_data[col_name] = normalized_values
    
    # Create a matrix of the smoothed data for correlation calculation
    smoothed_matrix = np.array([smoothed_data[col] for col in all_test_columns])
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(smoothed_matrix)
    
    # set all correlations below the threshold to 0
    corr_matrix[corr_matrix < corr_min] = 0
    
    # Convert correlation to distance (1 - correlation)
    # Higher correlation = smaller distance
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Apply K-means clustering to the distance matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # Count members in each cluster
    cluster_counts = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_counts:
            cluster_counts[label] = []
        cluster_counts[label].append((all_test_columns[i], i))  # Store column name and index
    
    # Create a mapping from column name to index in the correlation matrix
    col_to_idx = {col: i for i, col in enumerate(all_test_columns)}
    
    # Filter and refine clusters to ensure all members have correlation >= corr_min with each other
    valid_clusters = {}
    for label, members_with_idx in cluster_counts.items():
        if len(members_with_idx) < min_cluster_size:
            continue  # Skip clusters that are too small
            
        # Extract just the column names for easier processing
        members = [m[0] for m in members_with_idx]
        
        # Iteratively remove members until all remaining members have correlation >= corr_min with each other
        while len(members) >= min_cluster_size:
            # Check if all pairs have correlation >= corr_min
            all_above_threshold = True
            lowest_avg_corr = float('inf')
            member_to_remove = None
            
            # Calculate average correlation for each member with all other members
            avg_corrs = {}
            for i, member1 in enumerate(members):
                idx1 = col_to_idx[member1]
                total_corr = 0
                below_threshold_count = 0
                
                for member2 in members:
                    if member1 == member2:
                        continue
                    idx2 = col_to_idx[member2]
                    if corr_matrix[idx1, idx2] < corr_min:
                        below_threshold_count += 1
                    total_corr += corr_matrix[idx1, idx2]
                
                # If any member has correlations below threshold with others, the cluster doesn't satisfy the condition
                if below_threshold_count > 0:
                    all_above_threshold = False
                    avg_corr = total_corr / (len(members) - 1)
                    avg_corrs[member1] = avg_corr
                    if avg_corr < lowest_avg_corr:
                        lowest_avg_corr = avg_corr
                        member_to_remove = member1
            
            # If all pairs have correlation >= corr_min, we're done with this cluster
            if all_above_threshold:
                break
                
            # Otherwise, remove the member with the lowest average correlation
            if member_to_remove:
                members.remove(member_to_remove)
            else:
                # Shouldn't happen, but just in case
                break
        
        # Only keep clusters that still have enough members
        if len(members) >= min_cluster_size:
            # If the cluster is too large, select the members with highest average correlation
            if len(members) > max_cluster_size:
                # Calculate average correlation for each member with all other members
                avg_corrs = {}
                for member in members:
                    idx1 = col_to_idx[member]
                    total_corr = 0
                    for other_member in members:
                        if member == other_member:
                            continue
                        idx2 = col_to_idx[other_member]
                        total_corr += corr_matrix[idx1, idx2]
                    avg_corrs[member] = total_corr / (len(members) - 1)
                
                # Sort members by average correlation (highest first)
                sorted_members = sorted(members, key=lambda m: avg_corrs[m], reverse=True)
                members = sorted_members[:max_cluster_size]
            
            valid_clusters[label] = members
    
    print(f"Found {len(valid_clusters)} clusters with at least {min_cluster_size} members:")
    for label, members in valid_clusters.items():
        print(f"  Cluster {label}: {len(members)} members")
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Create a color palette for clusters
    # cluster_palette = sns.color_palette("husl", len(valid_clusters))
    
    # get palette without reds
    cluster_palette = sns.color_palette("husl", len(valid_clusters))
    # rotate the pallete 3 times
    n = 4
    cluster_palette = cluster_palette[n:] + cluster_palette[:n]
    
    for k,v in valid_clusters.items():
        # get first value in the cluster
        Y = normalized_data[v[0]] * 0
        break
    
    # Plot each cluster with its own color
    ymin,ymax = np.inf, -np.inf
    N = 0
    for i, (cluster_label, members) in enumerate(valid_clusters.items()):
        # Skip clusters not in selected_clusters if it's specified
        if selected_clusters is not None and cluster_label not in selected_clusters:
            continue
            
        cluster_color = cluster_palette[i]
        
        # generate random rescaling factor for each cluster
        f = np.random.uniform(0.9, 1.1)
        
        # Plot each member of the cluster
        for j, col_name in enumerate(members):
            y = normalized_data[col_name] * f  # Rescale the y values
            jitter = np.random.uniform(-0.02, 0.02)
            y = y + jitter
            
            # add to average
            Y += y
            N += 1
            
            # Ensure ymin, ymax are updated
            ymin = min(ymin, np.min(y))
            ymax = max(ymax, np.max(y))
            
            # Use the same color for all members of the cluster, with slight alpha variations
            alpha = 0.7 + 0.3 * (j / len(members))  # Vary alpha between 0.7 and 1.0
            plt.plot(X_feats, y, linewidth=2, 
                    color=cluster_color, alpha=alpha, 
                    label=f"{col_name} (Cluster {cluster_label})")
    
    # Set plot properties with seaborn styling
    plt.xlabel('Model Checkpoint (Global Step)', fontsize=18)
    plt.ylabel('Performance', fontsize=18)
    
    # compute average of all members in the cluster
    Y /= N
    # Plot the average of the cluster members in red dashed line
    plt.plot(X_feats, Y, linewidth=3, 
            #  linestyle='--', 
            linestyle='--',
             color='red', 
            label=f'Average', 
            alpha=0.8)
    
    # Update title to reflect if specific clusters are being shown
    plt.title(f'Benchmark Learning Curves', fontsize=20)
    # if selected_clusters is not None:
    #     clusters_str = ', '.join(map(str, selected_clusters))
    #     plt.title(f'Clustered Performance Curves - Clusters [{clusters_str}]', fontsize=16)
    # else:
    #     plt.title(f'Clustered Performance Curves (Epanechnikov Kernel, {n_clusters} clusters)', fontsize=16)

    
    # Add legend with cluster grouping
    if show_legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_cluster = {}
        for handle, label in zip(handles, labels):
            cluster = label.split('(Cluster ')[1].split(')')[0]
            if cluster not in by_cluster:
                by_cluster[cluster] = []
            by_cluster[cluster].append((handle, label))
        
        # Create a legend with cluster grouping
        legend_handles = []
        legend_labels = []
        for cluster, items in by_cluster.items():
            for handle, label in items:
                legend_handles.append(handle)
                legend_labels.append(label)
            # Add a separator between clusters (empty entry)
            if cluster != list(by_cluster.keys())[-1]:  # If not the last cluster
                legend_handles.append(plt.Line2D([0], [0], color='white'))
                legend_labels.append('')
        
        plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                fontsize=16, frameon=True, framealpha=0.7)
    else:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h for h, l in zip(handles, labels) if 'Average' in l]
        labels = [l for l in labels if 'Average' in l]
        # make the legend box inside the plot boundary, upper right
        plt.legend(handles, labels, loc='upper right', fontsize=16, frameon=True, framealpha=0.7)
        
        # create legend showing only average
        # plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5), 
        #         fontsize=10, frameon=True, framealpha=0.7)
        # plt.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.7)
    
    # Calculate the actual min and max values across all normalized data
    # all_min = min(np.min(values) for values in normalized_data.values())
    # all_max = max(np.max(values) for values in normalized_data.values())
    
    # Add a small margin to the limits
    margin = 0.02
    plt.ylim(ymin - margin, ymax + margin)
    
    # Add tight layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'clustered_columns_plot_{rand_seed}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    # create_test_plots()
    
    # Example using default Gaussian smoothing
    # plot_multiple_columns(7)
    
    # Example using spline smoothing with original values shown
    # plot_multiple_columns(5, smooth_method='spline', smooth_factor=10, show_original=True)
    
    # plot_multiple_columns(1, smooth_method='kernel', smooth_factor=10, show_original=True)
    plot_multiple_columns(1, smooth_method='gaussian', smooth_factor=3, show_original=True)
