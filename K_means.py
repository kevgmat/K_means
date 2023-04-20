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
        self.coordinate_1 = np.array(dataframe['coordinate1'].tolist())
        self.coordinate_2 = np.array(dataframe['coordinate2'].tolist())
        self.coordinate_3 = np.array(dataframe['coordinate3'].tolist())
        self.coordiante_4 = np.array(dataframe['coordinate4'].tolist())
        self.coordiante_5 = np.array(dataframe['coordinate5'].tolist())

        self.input_data = []
        self.null_data = []

        for i in range(0, len(self.coordinate_1)):
            if self.coordinate_1[i] == 0 or self.coordinate_2[i] == 0:
                self.null_data.append([self.coordinate_1[i], self.coordinate_2[i]])
            else:
                self.input_data.append([self.coordinate_1[i], self.coordinate_2[i]])

        self.input_data = np.array(self.input_data)

        plt.figure(figsize = (16,5))
        plt.subplot(1,2,1)
        sns.histplot(dataframe['coordinate1'])
        plt.subplot(1,2,2)
        sns.histplot(dataframe['coordinate2'])
        plt.show()
        sns.boxplot(dataframe['coordinate2'])

        coordinate_1_ptile_25 = dataframe['coordinate1'].quant
        coordinate_1_ptile_75 = dataframe['coordinate1'].quant

        upper_limit = coordinate_1_ptile_75 -1.5*iqr

        new_df = dataframe[dataframe['coordinate1']<upper_limit]
        print(new_df.shape)






    def get_data(self):
        return self.input_data

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
print(input.shape)

model = KMean()
model.predict(input)


# sample_data = np.random.randn(100,2)
# print(sample_data)
# print(type(sample_data))
# model = KMean()
# model.predict(sample_data)


# import pandas as pd
#
# # Replace "file.csv" with the name of your CSV file
# df = pd.read_csv("test.csv")
#
# # Check if any columns have empty values
# if df.isnull().values.any():
#     print("The CSV file has empty values.")
# else:
#     print("The CSV file does not have any empty values.")