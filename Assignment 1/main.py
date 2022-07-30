from decimal import Decimal
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.core import Flatten
tf.__version__

dataset = pd.read_csv('input2.csv', delimiter=';', header=None)
X = dataset.iloc[:, :28].values
y = dataset.iloc[:, 28].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X = sc.fit_transform(X)
y_train = np.array(y_train).astype('float32').reshape((-1, 1))
y_test = np.array(y_test).astype('float32').reshape((-1, 1))

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=28))

ann.add(tf.keras.layers.Dense(units=48, activation='relu'))

#ann.add(tf.keras.layers.Dense(units=10, activation='tanh'))

#ann.add(tf.keras.layers.Dense(units=36, activation='relu'))

#ann.add(tf.keras.layers.Dense(units=25, activation='relu'))

ann.add(tf.keras.layers.Dense(units=10, activation='softmax'))

ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ann1 = ann.fit(X_train, y_train, validation_data=(
    X_test, y_test), epochs=50, verbose=0)

# plt.plot(ann1.history['accuracy'])
# plt.plot(ann1.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# plt.plot(ann1.history['loss'])
# plt.plot(ann1.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

datasetPredict = pd.read_csv('inputPredict.csv', delimiter=';', header=None)
X_p = datasetPredict.iloc[:, :28].values
y_p = datasetPredict.iloc[:, 28].values


def resultVectorNulling():
    result = []
    for i in range(10):
        result.append(0)
    return result


loss, accuracy = ann.evaluate(X_test, y_test)
print(f'Loss is: {loss}')
print(f'Accuracy is: {accuracy}')

result = resultVectorNulling()
predict = ann.predict(X_p)

for i in range(0, len(predict)):
    result[np.argmax(predict[i])] = 1
    val = np.float32(predict[i][np.argmax(predict[i])])
    accuracy_number = val.item()
    if(i == 10):
        print('Zasumenie: ')
    elif(i == 15):
        print('Porusenie: ')
    print('The result is probably: {} with accuracy: {:.2f} {:.4f} in vector form: {} and should be {} '.format(
        np.argmax(predict[i]), accuracy_number, np.float32(predict[i][4]), np.array(result), y_p[i]))
    result = resultVectorNulling()
