import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
import random as rn
from tqdm import tqdm
from random import shuffle
from zipfile import ZipFile
from PIL import Image

# Define lookup dictionaries
lookup = dict()
reverselookup = dict()
count = 0

# Populate lookup dictionaries
for i in range(10):
    folder_path = f'~/Download/leapgestrecog/leapGestRecog/0{i}/'
    for j in os.listdir(folder_path):
        if not j.startswith('.'):
            lookup[j] = count
            reverselookup[count] = j
            count += 1

# Load and preprocess images
x_data = []
y_data = []
IMG_SIZE = 150
datacount = 0

for i in range(10):
    for j in os.listdir(f'~/Download/leapgestrecog/leapGestRecog/0{i}/'):
        if not j.startswith('.'):
            count = 0
            for k in os.listdir(f'~/Download/leapgestrecog/leapGestRecog/0{i}/{j}/'):
                path = f'~/Download/leapgestrecog/leapGestRecog/0{i}/{j}/{k}'
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                arr = np.array(img)
                x_data.append(arr)
                count += 1

            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount += count

# Convert data to numpy arrays
x_data = np.array(x_data, dtype='float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)

# One-hot encode labels
y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, IMG_SIZE, IMG_SIZE, 1))
x_data = x_data / 255

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.25, random_state=42)

# Define and compile the CNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
          activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(10, activation="softmax"))

batch_size = 256
epochs = 100

# Define callbacks for training
checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=False, period=1)
earlystop = EarlyStopping(
    monitor='val_loss', min_delta=0.001, patience=30, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16,
                          write_graph=True, write_grads=True, write_images=False)
csvlogger = CSVLogger(filename="training_csv.log", separator=",", append=False)
reduce = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
callbacks = [checkpoint, tensorboard, csvlogger, reduce]

# Compile the model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model.save('/hand_gesture/model')
