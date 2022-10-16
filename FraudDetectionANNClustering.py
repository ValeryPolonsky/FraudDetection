import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('creditcard.csv')
dataset = dataset[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                   'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']]
dataset_frauded_0 = dataset.loc[dataset['Class'] == 0]
dataset_frauded_1 = dataset.loc[dataset['Class'] == 1]

X_0 = dataset_frauded_0.values
X_1 = dataset_frauded_1.values

# Training the K-Means model on the dataset
def GetClusters(X, X_means, max_cluster_size):
    kmeans = KMeans(n_clusters = 2, init = 'k-means++')
    y_kmeans = kmeans.fit_predict(X)
    
    for i in range(0,2):        
        if (len(X[y_kmeans == i]) > max_cluster_size):
            GetClusters(X[y_kmeans == i], X_means, max_cluster_size)
        else:
            cluster = 0
            while(cluster in X_means):
                cluster += 1
                
            X_means[cluster] = X[y_kmeans == i]
            
            cluster_column = []
            for j in range(0,len(X_means[cluster])):
                cluster_column.append(cluster) 
            cluster_column =  np.array(cluster_column).reshape(len(cluster_column),1)
            cluster_column = cluster_column.astype(float)
                    
            X_means[cluster] = np.append(X_means[cluster], cluster_column, axis = 1)
                      
X_means = dict()
GetClusters(X_0, X_means, max_cluster_size = 1000)

# Creating new dataset with clusters
new_dataset = pd.DataFrame()
max_cluster = 0
for cluster in X_means:
    new_dataset = pd.concat([new_dataset, pd.DataFrame(X_means[cluster])])
    
    if (cluster > max_cluster):
        max_cluster = cluster
 
cluster_1 = max_cluster + 1
cluster_column = []
for j in range(0,len(X_1)):
    cluster_column.append(cluster_1) 
cluster_column =  np.array(cluster_column).reshape(len(cluster_column),1)
cluster_column = cluster_column.astype(float)
X_1 = np.append(X_1, cluster_column, axis = 1)

new_dataset = pd.concat([new_dataset, pd.DataFrame(X_1)])
new_dataset = new_dataset.rename(columns={new_dataset.columns[0]: 'V1',
                                          new_dataset.columns[1]: 'V2',
                                          new_dataset.columns[2]: 'V3',
                                          new_dataset.columns[3]: 'V4',
                                          new_dataset.columns[4]: 'V5',
                                          new_dataset.columns[5]: 'V6',
                                          new_dataset.columns[6]: 'V7',
                                          new_dataset.columns[7]: 'V8',
                                          new_dataset.columns[8]: 'V9',
                                          new_dataset.columns[9]: 'V10',
                                          new_dataset.columns[10]: 'V11',
                                          new_dataset.columns[11]: 'V12',
                                          new_dataset.columns[12]: 'V13',
                                          new_dataset.columns[13]: 'V14',
                                          new_dataset.columns[14]: 'V15',
                                          new_dataset.columns[15]: 'V16',
                                          new_dataset.columns[16]: 'V17',
                                          new_dataset.columns[17]: 'V18',
                                          new_dataset.columns[18]: 'V19',
                                          new_dataset.columns[19]: 'V20',
                                          new_dataset.columns[20]: 'V21',
                                          new_dataset.columns[21]: 'V22',
                                          new_dataset.columns[22]: 'V23',
                                          new_dataset.columns[23]: 'V24',
                                          new_dataset.columns[24]: 'V25',
                                          new_dataset.columns[25]: 'V26',
                                          new_dataset.columns[26]: 'V27',
                                          new_dataset.columns[27]: 'V28',
                                          new_dataset.columns[28]: 'Amount',
                                          new_dataset.columns[29]: 'Class',
                                          new_dataset.columns[30]: 'Cluster'})
new_dataset = new_dataset.sample(n = len(new_dataset.index))
new_dataset.to_csv('creditcard_clusters.csv')


    


