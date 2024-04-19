import numpy as np
import datetime
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# generate 2d classification dataset
X, y = make_moons(n_samples=400, noise=0.1)

plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.xlabel('x1')
plt.ylabel('x2');

#Divide the dataset in design and test
X_design, X_test, y_design, y_test = train_test_split(X,  # predictors
    y,  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility
with tf.device("GPU:0"):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10, activation='tanh', input_shape=(2, )))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_design, y_design,batch_size=32, epochs=500)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))