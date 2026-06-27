# K-Means Clustering

### What is K-Means Clustering?
- Used to cluster data into K Clusters
- Its an unsupervised Learning Algorithm

### K-Means Clustering Algorithm

For a High-Level Intuition, choose K = 2

Below are steps that algorithms follows to cluster N-dimensional data into 2 clusters
1. Decide number of clusters to use - K = 2 (we decided above)
2. Scale our data (so it's in a common range). Data has to be >= 1. We scale our data in 1-10 range
3. Randomly assign K=2 centroid points in our data space.
4. Categorize our data points to the closest centroids it can see.
5. Once closest data points to our K=2 clusters are found, find new centroids based on our grouped datapoints
6. Update original K=2 centroids to the new centroids calculated
7. Repeat steps 3-6 until no data-points change cluster assignment - aka convergence

### How do we decide the optimal value of K?
1. Make Use of Elbow Method - Essentially, run all the above steps but with different K values from 1 to any number reasonably larger enough for our data to be grouped. 
2. plot K value against Total Sum of Squared Errors per cluster.
3. The best K is the one at the Elbow of the curve