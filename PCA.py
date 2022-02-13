import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class PCA:
    """
    Basic implementation of PCA algorithm.

    """
    def __init__(self, n_components):
        self.n_components = n_components


    # Before applying PCA, the variables will be standardized to have a mean
    # of 0 and a standard deviation of 1. This is important because all
    # variables go through the origin point
    # (where the value of all axes is 0) and share the same variance.
    def standardize_data(self, arr):
        """ This function standardize an array, subract it's mean value. and then
        divide the standard deviation. Works for 2D array.
        """
        rows, columns = arr.shape
        standardized_array = np.zeros(arr.shape)

        for column in range(columns):
            # calculate mean of each column in table
            mean = np.mean(arr[:, column])
            std = np.std(arr[:, column])
            # standardize every entry in column
            for row in range(rows):
                arr[row, column] = (arr[row, column] - mean)/std

        return arr

    def compute(self, arr):
        """
        """
        arr = self.standardize_data(arr)
        # minimizing reconstruction error is equivalent to maximizing the variance

        # calculate the covariance matrix (measure of how much each of the
        # dimensions varies from the mean with respect to each other)
        covariance_matrix = np.cov(arr.T)
        print(covariance_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        print("Eigenvector: \n",eigen_vectors,"\n")
        print("Eigenvalues: \n", eigen_values, "\n")
        # Calculating the explained variance on each of componentsvariance_explained = []
        variance_explained = []
        for i in eigen_values:
             variance_explained.append((i/sum(eigen_values))*100)
        print(variance_explained)

        # Identifying components that explain at least 95%
        cumulative_variance_explained = np.cumsum(variance_explained)
        print(cumulative_variance_explained)

        # Visualizing the eigenvalues and finding the "elbow" in the graphic
        sns.lineplot(x = [1,2,3,4], y=cumulative_variance_explained)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Explained variance vs Number of components")
        plt.show()
        # Using two first components (because those explain more than 95%)
        projection_matrix = (eigen_vectors.T[:][:2]).T
        print(projection_matrix)

        # Getting the product of original standardized X and the eigenvectors
        X_pca = arr.dot(projection_matrix)
        print(X_pca)

# Test function
if __name__ == "__main__" :
    pca = PCA(2)
    arr = np.asarray([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    std_arr = pca.standardize_data(arr)
    print(std_arr)
    p = pca.compute(arr)
    print(p)
