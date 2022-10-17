import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras_tuner as kt
import joblib
import math
from tensorflow import keras

# Importing the dataset
dataset = pd.read_csv('creditcard_clusters.csv')
dataset = dataset[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                   'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Cluster']]

dataset_merged = pd.DataFrame()
clusters = np.sort(dataset.Cluster.unique())
sample_size = 20
for cluster in clusters:
    dataset_cluster = dataset.loc[dataset['Cluster'] == cluster]
    
    dataset_cluster_sample_size = sample_size;
    if (len(dataset_cluster.index) < sample_size):
        dataset_cluster_sample_size = len(dataset_cluster.index)          
    
    dataset_cluster = dataset_cluster.sample(n = dataset_cluster_sample_size)
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
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(np.array(y).reshape(len(y),1)).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Accuracy ~ 88%
# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, 
                                    criterion = 'gini', 
                                    random_state = 0,
                                    verbose=True)
classifier.fit(X_train, y_train)

# Save classifier
classifier.save('ClassifierANN')

# Load classifier
classifier = keras.models.load_model("ClassifierANN")

# Predicting the Test set results
y_pred = classifier.predict(X_test)

cluster = 512
y_cluster = []
for i in range(0,len(y_test)):
    if (y_test[i][cluster] == 1):
        y_cluster.append(y_test[i])
y_cluster = np.array(y_cluster)  

# Making the Confusion Matrix
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
cm = multilabel_confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

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



