import numpy as np
import torch

def bayesian_std(y, Y, weight=None):
    """
    Calculate a stabilized standard deviation for small samples
    using a larger population as prior.
    
    Parameters:
    y (array-like): Small sample array
    Y (array-like): Larger population array
    weight (float): Weight for the prior (None = automatic weighting based on sample size)
    
    Returns:
    float: Stabilized standard deviation
    """
    n = len(y)
    N = len(Y)
    
    # Calculate standard deviations
    std_y = np.std(y, ddof=1) if n > 1 else 0
    std_Y = np.std(Y, ddof=1)
    
    # Automatic weighting based on sample size
    if weight is None:
        # As n increases, weight of prior decreases
        weight = 1 / (1 + n/5)  # Adjust the divisor to control how quickly prior influence fades
    
    # Weighted average of the two standard deviations
    return weight * std_Y + (1 - weight) * std_y


class Transform:
    """
    A class for data normalization transforms.
    
    Provides static methods for creating transform instances and
    instance methods for applying transforms and their inverses.
    """
    
    def __init__(self, transform_type, **params):
        """
        Initialize a transform instance.
        
        Args:
            transform_type (str): Type of transform ('normalize' or 'standardize')
            **params: Parameters required for the transform
        """
        self.transform_type = transform_type
        self.params = params
    
    @staticmethod
    def normalize(X, min_val=0, max_val=1):
        """
        Create a min-max normalization transform.
        
        Args:
            X (numpy.ndarray): Data to normalize
            min_val (float): Minimum value of the normalized range
            max_val (float): Maximum value of the normalized range
            
        Returns:
            tuple: (normalized_data, Transform instance)
        """
        X = np.asarray(X)
        X_min = np.min(X)
        X_max = np.max(X)
        
        # Create transform instance
        transform = Transform('normalize', 
                              X_min=X_min, 
                              X_max=X_max, 
                              min_val=min_val, 
                              max_val=max_val)
        
        # Apply transform
        X_norm = transform.run(X)
        
        return X_norm, transform
    
    @staticmethod
    def standardize(X, Y=None):
        """
        Create a standardization transform.
        
        Args:
            X (numpy.ndarray): Data to standardize
            
        Returns:
            tuple: (standardized_data, Transform instance)
        """
        if Y is not None:
            return Transform.task_standardize(X, Y)
        
        X = np.asarray(X)
        mu = np.mean(X)
        sigma = np.std(X)
        
        # Create transform instance
        transform = Transform('standardize', mu=mu, sigma=sigma)
        
        # Apply transform
        X_std = transform.run(X)
        
        return X_std, transform
    
    @staticmethod
    def task_standardize(X, Y):
        """
        Create task-wise standardization transform (one transform per task).
        """
        t = X[:,1].long()
        means, stds, ys = [], [], []
        m = t.max() + 1
        for i in range(m):
            y = Y[t==i][:,0]
            mu = y.mean()
            yc = y - mu
            ys.append(yc)
            means.append(mu)
        means = np.array(means)
        
        # calculate stds using the pooled population data (when sparse)
        yy = torch.cat(ys).numpy()
        stds = np.array([bayesian_std(y.numpy(), yy) for y in ys])
        
        # actually perform the standardization
        Y_std = Y.clone()
        for i in range(m):
            Y_std[t==i] = (Y[t==i] - means[i]) / stds[i]
            
        # return the standardized Y, and the means and stds for later inverse transform
        return Y_std, Transform('standardize', mu=means, sigma=stds)
    
    
    def run(self, X):
        """
        Apply the transform to the data.
        
        Args:
            X (numpy.ndarray): Data to transform
            
        Returns:
            numpy.ndarray: Transformed data
        """
        X = np.asarray(X)
        
        if self.transform_type == 'normalize':
            X_min = self.params['X_min']
            X_max = self.params['X_max']
            min_val = self.params['min_val']
            max_val = self.params['max_val']
            
            # Handle the case where min == max to avoid division by zero
            if X_min == X_max:
                return np.ones_like(X) * min_val
            
            return min_val + (max_val - min_val) * (X - X_min) / (X_max - X_min)
        
        elif self.transform_type == 'standardize':
            mu = self.params['mu']
            sigma = self.params['sigma']
            
            # Handle the case where std == 0 to avoid division by zero
            if sigma == 0:
                return np.zeros_like(X)
            
            return (X - mu) / sigma
        
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")
    
    def inv(self, X):
        """
        Apply the inverse transform to the data.
        
        Args:
            X (numpy.ndarray): Data to inverse transform
            
        Returns:
            numpy.ndarray: Original scale data
        """
        X = np.asarray(X)
        
        if self.transform_type == 'normalize':
            X_min = self.params['X_min']
            X_max = self.params['X_max']
            min_val = self.params['min_val']
            max_val = self.params['max_val']
            
            # Handle the case where min_val == max_val to avoid division by zero
            if min_val == max_val:
                return np.ones_like(X) * X_min
            
            return X_min + (X_max - X_min) * (X - min_val) / (max_val - min_val)
        
        elif self.transform_type == 'standardize':
            mu = self.params['mu']
            sigma = self.params['sigma']
            
            return X * sigma + mu
        
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")


# Example usage:
if __name__ == "__main__":
    # Create an array for testing
    X = np.array([1, 2, 3, 4, 5])
    
    # Min-max normalization
    X_norm, norm_transform = Transform.normalize(X)
    print("\nMin-Max Normalization:")
    print("Normalized data:", X_norm)
    print("Using transform.run():", norm_transform.run(X))
    print("Inverse transform:", norm_transform.inv(X_norm))
    
    # Custom range normalization
    X_norm_custom, norm_transform_custom = Transform.normalize(X, min_val=-1, max_val=1)
    print("\nCustom Range Normalization [-1, 1]:")
    print("Normalized data:", X_norm_custom)
    print("Using transform.run():", norm_transform_custom.run(X))
    print("Inverse transform:", norm_transform_custom.inv(X_norm_custom))
    
    # Standardization
    X_std, std_transform = Transform.standardize(X)
    print("\nStandardization:")
    print("Standardized data:", X_std)
    print("Using transform.run():", std_transform.run(X))
    print("Inverse transform:", std_transform.inv(X_std))
    
    # Verify relationships
    print("\nVerify relationships:")
    print("X_norm == norm_transform.run(X):", np.allclose(X_norm, norm_transform.run(X)))
    print("X == norm_transform.inv(X_norm):", np.allclose(X, norm_transform.inv(X_norm)))
    print("X_std == std_transform.run(X):", np.allclose(X_std, std_transform.run(X)))
    print("X == std_transform.inv(X_std):", np.allclose(X, std_transform.inv(X_std)))
