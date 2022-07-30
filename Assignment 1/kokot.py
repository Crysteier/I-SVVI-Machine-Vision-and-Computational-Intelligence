# Importing the libraries
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
tf.version

dataset_train = pd.read_csv('training.csv')
X_train = dataset_train.iloc[:, 10:39].values
y_train = dataset_train.iloc[:, :10].values

dataset_test = pd.read_csv('test.csv')
X_test = dataset_test.iloc[:, 10:39].values
y_test = dataset_test.iloc[:, :10].values

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Initialising the ANN
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=28, activation='relu'))
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=10))

# Compiling the ANN
ann.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])

# Fitting the ANN to the Training set
ann1 = ann.fit(X_train, y_train, validation_data=(
    X_test, y_test),  batch_size=32, epochs=10)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)
print(y_pred)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# print (cm)
accuracy_score(y_test, y_pred)
