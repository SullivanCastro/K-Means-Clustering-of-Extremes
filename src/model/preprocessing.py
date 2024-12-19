import numpy as np
from numpy.typing import ArrayLike


class Preprocessing():

    @staticmethod
    def transform_to_extreme_values(data: ArrayLike) -> ArrayLike:
        """
        Transform the data to extreme values
        
        Parameters
        ----------
        data : np.array of shape (N_SAMPLES, N_DIMS)
            The data to transform
            
        Returns
        -------
        np.array
            The transformed data
        """
        # Ensure input is a numpy array
        data = np.asarray(data)
        
        # Calculate the empirical cumulative distribution function (ECDF) per dimension
        n_samples, n_dims = data.shape
        transformed_data = np.zeros_like(data, dtype=float)
        
        # X = (1/(1-F1(X_1),...,1-Fn(X_n)))
        for dim in range(n_dims):
            # Sort data and compute ranks
            sorted_indices = np.argsort(data[:, dim])
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, n_samples + 1)
            
            # Compute ECDF
            ecdf = ranks / n_samples
            
            # Transform to extreme values
            transformed_data[:, dim] = 1 / (1 - ecdf + 1e-10)
        
        return transformed_data
    

    @staticmethod
    def filter_largest(data: ArrayLike, threshold: float) -> ArrayLike:
        """
        Filter the data to keep only the observations with the largest norms.
        
        Parameters
        ----------
        data : np.array of shape (N_SAMPLES, N_DIMS)
            The transformed data where norms will be calculated.
        threshold : float
            The percentage (0 < threshold <= 1) of data to retain based on the largest norms.
            
        Returns
        -------
        np.array
            The filtered data containing only the top threshold percentage of observations.
        """
        if not (0 < threshold <= 1):
            raise ValueError("Threshold must be a float between 0 and 1.")
        
        # Compute the norm of each observation
        norms = np.linalg.norm(data, axis=1)
        
        # Determine the cutoff value for the top `threshold` percentage
        cutoff = np.percentile(norms, 100 * (1 - threshold))
        
        # Filter the data to keep only observations with norms above the cutoff
        filtered_data = data[norms > cutoff]
        
        return filtered_data


    @staticmethod
    def project_onto_unit_sphere(data):
        return data / np.linalg.norm(data, ord=1, axis=1, keepdims=True)


    @staticmethod
    def process(data, threshold=0.05):
        data = Preprocessing.transform_to_extreme_values(data)
        data = Preprocessing.filter_largest(data, threshold)
        data = Preprocessing.project_onto_unit_sphere(data)
        return data