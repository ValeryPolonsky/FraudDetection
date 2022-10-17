import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner as kt
import joblib
import math
import typing as typ
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

class ClassifierData():
    X_test: []
    X_train: []
    y_train: []
    y_test: []
    y_pred: []
    X_scaler: StandardScaler 
    y_encoder: OneHotEncoder
    Classifier: DecisionTreeClassifier
    ConfusionMatrix: multilabel_confusion_matrix
    AccuracyScore: accuracy_score
   
# Importing the dataset
dataset = pd.read_csv('creditcard_clusters.csv')
dataset = dataset[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                   'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class','Cluster']]

# Training the Decision Tree Classification model on the Training set
def CreateClusterDataSet(legit_cluster, frauded_cluster, sample_size):
    dataset_merged = pd.DataFrame()

    for cluster in [legit_cluster, frauded_cluster]:
        dataset_cluster = dataset.loc[dataset['Cluster'] == cluster]
        
        dataset_cluster_sample_size = sample_size;
        if (len(dataset_cluster.index) < sample_size):
            dataset_cluster_sample_size = len(dataset_cluster.index)          
        
        dataset_cluster = dataset_cluster.sample(n = dataset_cluster_sample_size)
        dataset_cluster = dataset_cluster[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                           'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                           'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']]        
        dataset_merged = pd.concat([dataset_merged, dataset_cluster])
        
    dataset_merged = dataset_merged.sample(frac=1).reset_index(drop=True)
    
    X = dataset_merged.iloc[:, :-1].values
    y = dataset_merged.iloc[:, -1].values
    
    # Taking care of missing data
    from sklearn.impute import SimpleImputer
    imputer_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer_numeric.fit(X)
    X = imputer_numeric.transform(X)
    
    # Encode categorical data
    # =============================================================================
    #     enc = OneHotEncoder()
    #     y = enc.fit_transform(np.array(y).reshape(len(y),1)).toarray()
    # =============================================================================
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifierData = ClassifierData()
    classifierData.X_train = X_train
    classifierData.X_test = X_test
    classifierData.y_train = y_train
    classifierData.y_test = y_test
    classifierData.X_scaler = sc
    #classifierData.y_encoder = enc
    
    return classifierData
    

classifierDataDict = dict()
cluster_frauded = 512  
clusters_legit = np.sort(dataset.Cluster.unique())
clusters_legit = np.delete(clusters_legit, np.where(clusters_legit==cluster_frauded))

for cluster_legit in clusters_legit: 
    classifierDataDict[cluster_legit] = CreateClusterDataSet(cluster_legit, cluster_frauded, 1000)

for cluster in classifierDataDict:
    classifier = DecisionTreeClassifier(criterion = 'gini',
                                        splitter='best',
                                        random_state = 0)
    classifier.fit(classifierDataDict[cluster].X_train, classifierDataDict[cluster].y_train)
    classifierDataDict[cluster].Classifier = classifier

# Predicting the Test set results
for cluster in classifierDataDict:
    classifierDataDict[cluster].y_pred = classifierDataDict[cluster].Classifier.predict(classifierDataDict[cluster].X_test)

# Making the Confusion Matrix
for cluster in classifierDataDict:
    classifierDataDict[cluster].ConfusionMatrix = multilabel_confusion_matrix(classifierDataDict[cluster].y_test, classifierDataDict[cluster].y_pred)
    classifierDataDict[cluster].AccuracyScore = accuracy_score(classifierDataDict[cluster].y_test, classifierDataDict[cluster].y_pred)
  
avg_cluster_accuracy = 0
for cluster in classifierDataDict:
    avg_cluster_accuracy += classifierDataDict[cluster].AccuracyScore
avg_cluster_accuracy = avg_cluster_accuracy/len(classifierDataDict)
    
# Predicting results by using all clusters
cluster_to_test = 1
dataset_cluster = dataset.loc[dataset['Cluster'].isin([cluster_to_test])]
dataset_cluster = dataset_cluster[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                   'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class']]
X_test = dataset_cluster.iloc[:, :-1].values
y_test = dataset_cluster.iloc[:, -1].values
y_pred_final_arr = []

for row in range(0,len(X_test)):
    y_pred_0_counter = 0
    y_pred_1_counter = 0
    y_pred_final = 0
    X_test_row = X_test[row,:]
    X_test_row = np.array(X_test_row).reshape(1,len(X_test_row))
    
    for cluster in classifierDataDict:       
        X_test_scaled = classifierDataDict[cluster].X_scaler.transform(X_test_row)       
        y_pred = classifierDataDict[cluster].Classifier.predict(X_test_scaled)
        y_pred_0_counter += len(y_pred[y_pred == 0])
        y_pred_1_counter += len(y_pred[y_pred == 1])
        
    if (y_pred_0_counter < 2):
        y_pred_final = 1
        
    y_pred_final_arr.append(y_pred_final)
    
y_pred_final_arr = np.array(y_pred_final_arr)

from sklearn.metrics import confusion_matrix, accuracy_score
cm_final = confusion_matrix(y_test, y_pred_final_arr)
acc_score_final = accuracy_score(y_test, y_pred_final_arr)
        




print('y_pred_0_counter: {0}'.format(y_pred_0_counter/len(dataset_cluster.index)))
print('y_pred_1_counter: {0}'.format(y_pred_1_counter/len(dataset_cluster.index)))
print('y_pred_final: {0}'.format(y_pred_final))












# Applying k-Fold Cross Validation
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion': ['gini'], 
               'splitter': ['best'],
               'min_samples_split': [2],
               'min_samples_leaf': [1,2,3],
               'min_weight_fraction_leaf': [0],
               'min_impurity_decrease': [0],
               'ccp_alpha': [0],
               'random_state': [0]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)



