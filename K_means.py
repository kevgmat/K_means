import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv(r'ieeedata.csv')

coordinate_1 = df['coordinate1'].tolist()
coordinate_2 = df['coordinate2'].tolist()
coordinate_3 = df['coordinate3'].tolist()
coordinate_4 = df['coordinate4'].tolist()
coordinate_5 = df['coordinate5'].tolist()

class Data_arrangement():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.coordinate_1 = dataframe['coordinate1'].tolist()
        self.coordinate_2 = dataframe['coordinate2'].tolist()
        self.coordinate_3 = dataframe['coordinate3'].tolist()
        self.coordinate_4 = dataframe['coordinate4'].tolist()
        self.coordinate_5 = dataframe['coordinate5'].tolist()
        self.main_input_list = []
        for i in range(0, len(coordinate_1)):
            self.main_input_list.append([self.coordinate_1[i],
                                         self.coordinate_2[i],
                                         self.coordinate_3[i],
                                         self.coordinate_4[i],
                                         self.coordinate_5[i]])
        # print(self.main_input_list)

        self.null_data = []
        self.removed_zeroes = []
        for i in range(0, len(self.main_input_list)):
            if 0 in self.main_input_list[i]:
                self.null_data.append(self.main_input_list[i])
            else:
                self.removed_zeroes.append(self.main_input_list[i])

        self.cleaned_data = self.clean(self.removed_zeroes)
        self.cleaned_data = np.array(self.cleaned_data)

    def clean(self, main_input_list):
        v = [self.coordinate_1, self.coordinate_2, self.coordinate_3, self.coordinate_4, self.coordinate_5]
        clean = [[],[],[],[],[]]
        for i in range(0, 5):
            q1, q3 = np.percentile(v[i], [25,60])
            upper_bound = q3 + 1.5*(q3- q1)
            clean[i] = [x for x in main_input_list if x[i] <= upper_bound]

        unique = []
        for item in clean[0]:
            if item not in unique:
                unique.append(item)

        # Next, check if each unique element in the first list is also in the other four lists
        intersection = []
        for item in unique:
            if item in clean[1] and item in clean[2] and item in clean[3] and item in clean[4]:
                intersection.append(item)
        return intersection

    def get_data(self):
        return self.cleaned_data

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMean():

    def __init__(self, K = 5, max_iterations = 100, steps= True):
        self.K = K
        self.max_iterations = max_iterations
        self.steps = steps

        self.clusters = [[] for i in range(self.K)]
        self.centroids = []

    def predict(self, input_data):
        self.input_data = input_data
        self.samples, self.dimensions = input_data.shape

        random_indexes = np.random.choice(self.samples, self.K, replace = False)
        self.centroids = [self.input_data[index] for index in random_indexes]

        for i in range(self.max_iterations):
            self.clusters = self.create_clusters(self.centroids)

            if self.steps:
                self.plot()

            old_centroids = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            if self.is_converged(old_centroids, self.centroids):
                break
            if self.steps:
                self.plot()

        return self.get_cluster_labels(self.clusters)



    def get_cluster_labels(self, clusters):
        labels = np.empty(self.samples)

        for cluster_index, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_index
        return labels

    def create_clusters(self, centroids):
        clusters = [[] for i in range(self.K)]
        for index, sample in enumerate(self.input_data):
            centroid_index = self.closest_centroid(sample, centroids)
            clusters[centroid_index].append(index)
        return clusters

    def closest_centroid(self, sample, centroids):
        distances = [distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.dimensions))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.input_data[cluster], axis = 0)
            centroids[cluster_index] = cluster_mean
        return centroids

    def is_converged(self, old_centroids, centroids):
        distances = [distance(old_centroids[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):

        fig, ax = plt.subplots(figsize = (12,8))

        for i, index in enumerate(self.clusters):
            point = self.input_data[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker = "x", color = 'black', linewidth = 2)
        plt.show()



test = Data_arrangement(df)
input= test.get_data()
input = input[:,[0,1]]
print(input)
print(input.shape)

model = KMean()
model.predict(input)
