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
                   'V21','V22','V23','V24','V25','V26','V27','V28','Amount','Class','Cluster']]
dataset_frauded_0 = dataset.loc[dataset['Class'] == 0]
dataset_frauded_1 = dataset.loc[dataset['Class'] == 1]

#sample_size = 0.00173
sample_size = 1
dataset_frauded_0_sample_size = int(len(dataset.index) * sample_size)
dataset_frauded_1_sample_size = int(len(dataset.index) * sample_size)

if (len(dataset_frauded_0) < dataset_frauded_0_sample_size):
    dataset_frauded_0_sample_size = len(dataset_frauded_0)
    
if (len(dataset_frauded_1) < dataset_frauded_1_sample_size):
    dataset_frauded_1_sample_size = len(dataset_frauded_1)

dataset_frauded_0 = dataset_frauded_0.sample(n = dataset_frauded_0_sample_size)
dataset_frauded_1 = dataset_frauded_1.sample(n = dataset_frauded_1_sample_size)

dataset_merged = pd.concat([dataset_frauded_0, dataset_frauded_1])
dataset_merged = dataset_merged.sample(frac=1).reset_index(drop=True)

X = dataset_merged.iloc[:, :-1].values
y = dataset_merged.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer_numeric = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_numeric.fit(X)
X = imputer_numeric.transform(X)

imputer_constant = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imputer_constant.fit(y[:,np.newaxis])
y = imputer_constant.transform(y[:,np.newaxis]).flatten()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the ANN model on the Training set
# Accuracy: 99.00 %
def CreateClassifier(X_train, y_train):
    classifier = keras.models.Sequential()
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=6, activation='relu'))
    classifier.add(keras.layers.Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = int(math.sqrt(len(X_train))), epochs = 200, verbose=1)
    return classifier

classifier = CreateClassifier(X_train, y_train)

# Save classifier
classifier.save('ClassifierANN')

# Load classifier
classifier = keras.models.load_model("ClassifierANN")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cvscores = []

for train, test in kfold.split(X_train, y_train):
	model = CreateClassifier(X_train[train], y_train[train])
	scores = model.evaluate(X_train[test], y_train[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
 
print("Accuracy: {:.2f} %".format(np.mean(cvscores)))
print("Standard Deviation: {:.2f} %".format(np.std(cvscores)))

# Searching the best parameters
def model_builder(hp):
  model = keras.models.Sequential()
  hp_units = hp.Int('units', min_value=2, max_value=10, step=1)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(units=1, activation='sigmoid'))
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


