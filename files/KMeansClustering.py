import numpy as np
class KMeansClustering:
    def __init__(self):
        self.x = None
        self.idx = None
        self.centroids = None
        self.k = None
        self.epochs_limit = None
        self.epochs_count = None
    def find_nearest_clusters(self):
        self.idx = np.zeros(self.x.shape[0])
        for i in range(self.x.shape[0]):
            min_id = 0
            mini = np.linalg.norm(self.x[i] - self.centroids[0])
            for j in range(self.k):
                if min(np.linalg.norm(self.x[i] - self.centroids[j]),mini) == np.linalg.norm(self.x[i] - self.centroids[j]):
                    min_id = j
                    mini = np.linalg.norm(self.x[i] - self.centroids[j])
            self.idx[i] = min_id
    def compute_centroids(self):
        self.centroids[0] = np.mean(self.x ,axis = 0)
        for i in range(self.k):
            points_in_cluster = self.x[self.idx == i]
            self.centroids[i] = np.mean(points_in_cluster,axis = 0)
    def fit(self,x,k,epochs = 100):
        self.x = x
        self.k = k
        self.epochs_limit = epochs
        self.epochs_count = 1
        self.centroids = np.zeros((self.k,self.x.shape[1]))
        while self.epochs_count <= self.epochs_limit:
            print("*"*(self.epochs_count//10),end = "\r")
            self.find_nearest_clusters()
            self.compute_centroids()
            self.epochs_count += 1
