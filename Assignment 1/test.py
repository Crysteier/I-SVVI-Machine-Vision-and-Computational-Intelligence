from sklearn.model_selection import train_test_split
import pandas as pd
from threading import active_count
import tensorflow as tf
from tensorflow.keras import models

mnist=tf.keras.datasets.mnist
dataset = pd.read_csv('input2.csv', delimiter=';', header=None)
X = dataset.iloc[:, :28]
y = dataset.iloc[:, 29:29]

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X)
print()
print(X_train)
print('------------')
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#model.add(tf.keras.layers.Dense(units=9, activation='relu', input_dim=28))

model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=3)

loss,accuracy=model.evaluate(X_test,y_test)
print(accuracy)
print(loss)

model.save('digits.model')
