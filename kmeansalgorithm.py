import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

epsilon = 0.0000001

def KmeansScikit(Data,clusters):
    # Initialize the class object
    kmeans = KMeans(n_clusters=clusters)

    # predict the labels of clusters.
    label = kmeans.fit_predict(Data)

    # Getting unique labels
    u_labels = np.unique(label)

    # plotting the results:
    for i in u_labels:
        plt.scatter(Data[label == i, 0], Data[label == i, 1], label=i)
    plt.legend()
    plt.show()

    # Getting the Centroids
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(label)

    # plotting the results:
    for i in u_labels:
        plt.scatter(Data[label == i, 0], Data[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    plt.show()

# Calculate the distances of two point in 2D array
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class MyKMeans(object):

    def __init__(self,K=3, max_iters=100, plot_steps=3):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # New centers which will be assigned in each iteration.
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self.assignSamples(self.centroids)

            if self.plot_steps > 0:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self.getCenterPoints(self.clusters)

            # check whether clusters changed
            if self.checkDelta(centroids_old, self.centroids):
                break

            if self.plot_steps > 0:
                self.plot()

            self.plot_steps = self.plot_steps - 1

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    # Assign each sample to the new cluster center
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def assignSamples(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.calculateClosestPointToCluster(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # Calculate the distances of each sample to new cluster center
    def calculateClosestPointToCluster(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    # Calculate new cluster center by calculating mean value of all points near to previous cluster center including it as well
    def getCenterPoints(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # Check whether there is no change while moving newly assigned cluster centers
    def checkDelta(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) <= epsilon

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()


def main():
    # Create data with 3 and 7 centers which means  there will be 3 or 7 clusters or any, it will return the Data and Cluster number
    Data,label = make_blobs(centers=7,n_samples=3000,n_features=2,shuffle=True,random_state=0)
    # Draw the created 2D data set
    plt.scatter(Data[:, 0], Data[:, 1],c='black')
    print(Data.shape)

    # Get cluster number while we entered during data creation.
    clusters = len(np.unique(label))
    print(clusters)

    # Run our own K-Means algorithm on Data set
    k = MyKMeans(K=clusters, max_iters=1500,plot_steps=3)
    k.predict(Data)
    k.plot()

    # Run original SCIKIT K-Means algorithm on Data set and plot results
    KmeansScikit(Data, clusters)


if __name__ == "__main__":
    main()