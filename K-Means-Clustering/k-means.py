"""
K-Means Clustering Implementation from Scratch
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import argparse
plt.ion()  # Turn on interactive mode for live plotting

class KMeans:

    def __init__(self, num_clusters=2, max_iterations=10):
        self.num_clusters = num_clusters        # number of clusters to group our data into
        self.max_iterations = max_iterations    # number of iterations to run the algorithm for
        self.centroids = np.array([])  # Initialize centroids as an empty array
        self.labels = np.array([])  # Initialize labels as an empty array
        self.sum_of_squared_distances = []  # Initialize empty list to hold SSE distances for various cluster sizes

    def normalize_data(self, X: np.ndarray) -> np.ndarray:
        """_summary_
        This function is used to nr
        Args:
            X (np.ndarray): The input data to be normalized

        Returns:
            np.ndarray: Normalized and scaled version of input data
        """
        min_scaled_value, max_scaled_value = 1, 10      # scaling our data to be between 1 to 10 value
        # X.min() is minimum value in our entire dataset
        # X.max() is maximum value in our entire dataset
        # Normalization says - Subtract minimum value from every data point and divide it by the spread (aka difference between max and min)
        X_normalized = (X - X.min()) / (X.max() - X.min())  # This normalizes our data in range 0 to 1
        # We need data to be in 1-10 range. So, 0 needs to be mapped to 1 and 1 needs to be mapped to 10
        # To do the above we use : normalized data * (desired max - desired min) + desired min
        # For eg: X_norm = 0, then : X_norm_scaled = 0 * (10-1) + 1 = 1     (our original value which was 0 is now mapped to 1)
        # Likewise, if X_norm = 1, then : X_norm_scaled = 1 * (10-1) + 1 = 10   (our original value which was 1 is now mapped to 10)
        # Any value in between 0-1 for X_norm now gets scaled between 1-10 which is our desired range of min-max values we want in our dataset
        X_norm_scaled = X_normalized * (max_scaled_value - min_scaled_value) + min_scaled_value     # scale our normalized 0-1 data to 1-10 range
        return X_norm_scaled    # Returns our data in 150x2 original dimensions

    def initialize_centroids_kmeanspp(self, X: np.ndarray, K: int):

        n = X.shape[0]
        centroids = np.empty((K, X.shape[1]))

        # randomly initialize first centroid
        centroids[0] = X[np.random.randint(n)]

        # Closest squared distance - this calculates squared distance from each data point to chose centroid
        closest_sq_dist = ((X - centroids[0]) ** 2).sum(axis=1)

        for c in range(1,K):
            probs = closest_sq_dist / closest_sq_dist.sum()
            next_idx = np.random.choice(n, p=probs)
            centroids[c] = X[next_idx]

            # calculate new squared distance wrt the new centroid
            new_sq_dist = ((X - centroids[c]) ** 2).sum(axis=1)
            closest_sq_dist = np.minimum(new_sq_dist, closest_sq_dist)

        self.centroids = centroids

    def initialize_centroids(self, X: np.ndarray, K: int):
        """_summary_
        X : 150x2 input matrix
        Args:
            X (np.ndarray): _description_
            K (int): _description_
        """
        # To find centroids for our 2D data, we need to find min and maximum value in each of our dimension
        # Once we find minimum and maximum, we randomly generate a number between these bounds to get our random centroids
        # X.min(axis=0) says in our X matrix, find minimum value along the rows since axis=0 is rows
        # X.max(axis=0) says in our X matrix, find maximum value along the rows since axis=0 is rows
        # For every column we get min-max value of that column and so we generate a random number for that column that sits in its min-max range
        # Our answer is - number_of_centroids (K) x dimension_of_our_data (aka columns of X)
        self.centroids = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=(K, X.shape[1]))

    def assign_clusters(self, X: np.ndarray):
        """_summary_
        X: 150x2 matrix

        Args:
            X (np.ndarray): _description_
        """
        # Our dataset is 150x2 but our centroids are 2x2. We cannot perform matrix operation right away
        # To assign clusters, we need each of the 150 data to be compared with the 2 centroids.
        # Expansion helps to map these operations in vectorized format
        # X[:,None,:] means add a empty dimension in 2nd axis - so 150x2 becomes 150x1x2
        # centroid[None,:,:] means add an empty dimension in the first axis - so 2x2 becomes 1x2x2
        # when we do X - centroids then - 150x1x2 (X) - 1x2x2 (centroids) becomes -> 150x2x2 (X) and 150x2x2 (centroids)
        # The 1 expands to match the dimension of the other array.
        # 150x2x2 = data_points x num_centroids x num_features (aka dimensions)
        X_expanded = X[:,None,:]
        centroids_expanded = self.centroids[None,:,:]   # 1x2x2
        # distances is now 150x2 but instead of it being data_points x features it is now, data_points x num_centroids
        # this means, our matrix now holds distance values of each data with each centroid
        distances = np.sqrt(((X_expanded - centroids_expanded) ** 2).sum(axis=2))   # sum along the features axis
        # Now, we find the centroids closest to our data_points. For that we find the minimum along the centroids axis
        # Hence, for each 150 data_points we get either 0 or 1 indicating 1st centroid or 2nd centroid that the data got clustered into
        self.labels = np.argmin(distances, axis=1)  # finding minimum distance centroid

    def update_centroids(self, X: np.ndarray, K: int):
        """_summary_
        X = 150x2 matrix

        Args:
            X (np.ndarray): _description_
            K (int): _description_
        """
        # We have 2 nested conditions here.
        # First is we loop over each centroid from all possible centroids
        # Next we ask, if any of this centroid is present in our labels list.
        # Labels list holds centroid assignments for each of our 150 data points
        # If none of the 150 data points got assigned any cluster, then we can't update our centroids, so we just take the previous centroid
        # Finally, if both conditions satisfy, we find all data_point rows which belong to our current centroid cluster.
        # Once all rows are found, we take the mean of the data along the rows, so this is our new centroid
        self.centroids = np.array([
            X[self.labels == j].mean(axis=0)
            if np.any(self.labels == j) else self.centroids[j]
            for j in range(K)
        ])  # this is still a num_centroids x num_features size matrix. So 2x2

    def calculate_sse(self, X: np.ndarray):
        # X = 150x2 matrix
        # centroids is 2x2 and labels is 150x1 array.
        # centroids[labels] says for each data_point in labels array, select that centroid which the label points to for that data_point
        # centroids[labels] ends up being 150x2 matrix too. Since labels is a list of [0, 1, 0, ...] and centroids in an array of 2 centroids
        # a label of 0 selects centroid #1 and label of 1 selects centroid #2
        diff = (X - self.centroids[self.labels])
        sse = (diff ** 2).sum()
        return sse

    def calculate_silhoutte_score(self, X: np.ndarray, K: int):
        """_summary_
        X = 150x2 matrix. Returns the mean silhouette score over all data points.
        Silhouette is undefined for K=1 (no "other" cluster to compare against), so we return 0.

        Args:
            X (np.ndarray): The (already normalized) input data
            K (int): The number of clusters
        """
        if K < 2:
            return 0.0

        n = X.shape[0]  # number of data points (150)

        # STEP 1: pairwise distance matrix D, shape n x n.
        # Same broadcasting trick as assign_clusters, but comparing X with itself.
        # X[:,None,:] is 150x1x2 and X[None,:,:] is 1x150x2 -> they broadcast to 150x150x2.
        # Summing the squared differences along the feature axis (axis=2) gives a 150x150 matrix
        # where D[i,j] is the Euclidean distance between point i and point j (and D[i,i] = 0).
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

        # STEP 2: one-hot label matrix M, shape n x K.
        # M[j,k] = 1 if point j belongs to cluster k, else 0.
        # np.eye(K) builds a KxK identity; indexing it by self.labels picks the matching row per point.
        M = np.eye(K)[self.labels]  # 150xK

        # count[k] = number of points in cluster k (column sums of the one-hot matrix)
        counts = M.sum(axis=0)  # length-K vector

        # STEP 3: sum of distances from each point to each whole cluster.
        # D @ M is n x K: entry [i,k] = sum over all j in cluster k of distance(i, j).
        sum_dist = D @ M  # 150xK

        # STEP 4: a(i) -> intra-cluster mean distance.
        # For each point grab the column of its OWN cluster, then divide by (count - 1) to exclude itself.
        own = self.labels  # cluster index per point
        own_sum = sum_dist[np.arange(n), own]              # sum of distances to own cluster
        own_counts = counts[own]                            # size of each point's own cluster
        # np.where guards against singleton clusters (count==1) which would divide by zero -> set a(i)=0
        a = np.where(own_counts > 1, own_sum / np.maximum(own_counts - 1, 1), 0.0)

        # STEP 5: b(i) -> mean distance to the NEAREST other cluster.
        # mean_dist[i,k] = average distance from point i to all points of cluster k.
        mean_dist = sum_dist / counts                       # 150xK, broadcast divide by per-cluster counts
        # Mask out each point's own cluster with +inf so it's never picked as the "nearest other" cluster.
        mean_dist[np.arange(n), own] = np.inf
        b = mean_dist.min(axis=1)                           # nearest other cluster's mean distance

        # STEP 6: per-point silhouette s(i) = (b - a) / max(a, b), then average over all points.
        # np.maximum is elementwise; guard the rare a==b==0 case to avoid 0/0.
        denom = np.maximum(a, b)
        s = np.where(denom > 0, (b - a) / denom, 0.0)
        return s.mean()

    def plot_clusters(self, X: np.ndarray, K: int, iteration: int):
        plt.clf()  # clear the existing figure instead of making a new one
        # Set the OS window title bar (separate from the in-plot title set by plt.title)
        plt.gcf().canvas.manager.set_window_title(f'K={K}')
        plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap='viridis', marker='o', edgecolor='k', s=100)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
        plt.title(f'Iteration {iteration}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.pause(0.5)  # Pause to update the plot

    def fit(self, X: np.ndarray, init="kmeans"):

        X_norm = self.normalize_data(X)     # 150x2 data
        for k in self.num_clusters:
            if init == "kmeans++":
                self.initialize_centroids_kmeanspp(X_norm, k)
            else:
                self.initialize_centroids(X_norm, k)
            old_centroids = np.array([])
            iteration = 1
            fig = plt.figure(figsize=(8, 6))

            while iteration <= self.max_iterations and not np.array_equal(self.centroids, old_centroids):
                old_centroids = self.centroids.copy()
                self.assign_clusters(X_norm)
                self.update_centroids(X_norm, k)
                self.plot_clusters(X_norm, k, iteration)
                iteration += 1

            sse = self.calculate_sse(X_norm)
            self.sum_of_squared_distances.append(sse)
            print(f"K={k}, SSE={sse}")

            plt.close(fig)  # close this K's figure before moving to the next K

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot

        return self.sum_of_squared_distances

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--init", help="Type of k-means initialization", default="kmeans")
    args = vars(ap.parse_args())

    k_range = range(1, 11)
    # load the iris dataset
    iris = load_iris()
    X = iris.data[:,:2]
    print(f"Running K-means with {args["init"]} initialization")
    kmeans = KMeans(num_clusters=k_range, max_iterations=10)
    sum_of_squared_distances = kmeans.fit(X, init=args["init"])

    # Plot the elbow curve: SSE vs K. The "elbow" suggests a good choice of K.
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), sum_of_squared_distances, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
    sys.exit(0)
