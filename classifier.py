from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import re
import numpy as np

# Load data
oscillators = pd.read_csv("/home/hellsbells/Desktop/oscillator_jacobains.csv")
notOsc = pd.read_csv("/home/hellsbells/Desktop/non-oscillator_jacobians.csv")

osc = []
nonOsc = []

def convert_jac_string(j):
    j = re.sub('\s{1,}', '$', j)
    j = j[1:-1]
    if j.startswith("$"):
        j = j[1:]
    if j.endswith("$"):
        j = j[:-1]
    j = j.split("$")

    newJ = []
    for num in j:
        if 'e' in num:
            if '+' in num:
                num, e = num.split('e')
                e = e[1:]
                newJ.append(float(num) * 10 ** (float(e)))
            elif '-' in num:
                num, e = num.split('e')
                e = e[1:]
                newJ.append(float(num) * 10 ** (-1 * float(e)))
        else:
            newJ.append(float(num))
    return newJ

# Hacky bullshit to convert strings into list of floats
for i in range(len(oscillators["jacobian"])):
    try:
        j = oscillators["jacobian"][i]
        j = convert_jac_string(j)
        osc.append(j)
    except Exception as e:
        print(e)

for i in range(len(notOsc["jacobian"])):
    try:
        j = notOsc["jacobian"][i]
        j = convert_jac_string(j)
        nonOsc.append(j)
    except Exception as e:
        print(e)

j_train = []
label_train = []
j_test = []
label_test = []



for i in range(len(osc)):
    j_test.append(osc[i])
    label_test.append(True)

for i in range(1000):
    j_train.append(nonOsc[i])
    label_train.append(False)

for i in range(1000, len(nonOsc)):
    j_test.append(nonOsc[i])
    label_test.append(False)

datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0, # Randomly zoom image
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
#
datagen.fit(j_train)

# model = Sequential()
# opt = Adam(lr=0.0001)
# model.compile(optimizer=opt,loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
#
# history = model.fit(j_train,label_train,epochs = 500 , validation_data = (j_test, label_test))
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(500)
#
# plt.figure(figsize=(15, 15))
# plt.subplot(2, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(2, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
