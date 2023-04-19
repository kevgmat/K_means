import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r'ieeedata.csv')

coordinate_1 = df['coordinate1'].tolist()
coordinate_2 = df['coordinate2'].tolist()
coordinate_3 = df['coordinate3'].tolist()
coordinate_4 = df['coordinate4'].tolist()
coordinate_5 = df['coordinate5'].tolist()

print(coordinate_1)

def distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMean():

    def __init__(self, K = 5, max_iterations = 100, steps= True):
        self.K = K
        self.max_iterations = max_iterations
        self.steps = steps

        self.clusters = [[] for i in range(self.K)]
        self.centroids = []



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