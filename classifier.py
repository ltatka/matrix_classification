from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np

print("Done")
#
# query = {"num_nodes": 3, "oscillator": True}
# result = mm.query_database(query)
#
# jacobians = []
# model_ids = []
# oscillator = []
# for i in range(10):
#     model = result[i]["model"]
#     m = te.loada(model)
#     model_ids.append(result[i]["ID"])
#     j = m.getFullJacobian()
#     jacobians.append(j)
#     oscillator.append(result[i]["oscillator"])
#
# query = {"num_nodes": 3, "oscillator": False}
# result = mm.query_database(query)
#
# for i in range(10):
#     model = result[i]["model"]
#     m = te.loada(model)
#     model_ids.append(result[i]["ID"])
#     j = m.getFullJacobian()
#     jacobians.append(j)
#     oscillator.append(result[i]["oscillator"])
#
# data = {"oscillator": oscillator,
#         "ID": model_ids,
#         "jacobian": jacobians}
#
