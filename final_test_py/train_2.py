import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
import pickle

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils

from sklearn.utils import shuffle

# train a bunch of model sequentially at once

print('-----------------------------------')
print('load data')
print('-----------------------------------')

# load dataset
f = open(r'../ships-in-satellite-imagery/shipsnet_v2.json')
dataset = json.load(f)
f.close()

df = pd.DataFrame(dataset)
new_df = df.drop(['locations','scene_ids'], axis=1)

## shuffle data
new_df = shuffle(new_df)
data = list(new_df.data)
data = np.array(data).astype('uint8')
labels = list(new_df['labels'])
labels = np.array(labels).astype('uint8')

## Normalized pixel values
x = data / 255.

## Create training set
x = x.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])
y = to_categorical(labels, num_classes=2)

print('-----------------------------------')
print('TRAINING START')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 12')
print('-----------------------------------')


## Train models

## model 12 

model_12 = Sequential()
model_12.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_12.add(MaxPooling2D(pool_size=(2, 2)))

model_12.add(Dropout(0.5))

model_12.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_12.add(MaxPooling2D(pool_size=(2, 2)))

model_12.add(Dropout(0.5))

model_12.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_12.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_12.add(MaxPooling2D(pool_size=(2, 2)))

model_12.add(Dropout(0.5))

model_12.add(Flatten())
model_12.add(Dense(512, activation='relu'))
model_12.add(Dropout(0.5))
model_12.add(Dense(2, activation='softmax'))

## compile a model
model_12.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_12 = model_12.fit(x, y, batch_size=20, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 12')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model12_hist.pkl', 'wb') as file_name:
    pickle.dump(history_12.history, file_name)

model_12.save('../cnn_model/classifer_1/model12.h5')

print('-----------------------------------')
print('end train model 12')
print('-----------------------------------')

print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 13')
print('-----------------------------------')


## model_13 train
model_13 = Sequential()
model_13.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_13.add(MaxPooling2D(pool_size=(2, 2)))

model_13.add(Dropout(0.5))

model_13.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_13.add(MaxPooling2D(pool_size=(2, 2)))

model_13.add(Dropout(0.5))

model_13.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
## ADD THIS DROPOUT
model_13.add(Dropout(0.5))
model_13.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_13.add(MaxPooling2D(pool_size=(2, 2)))

model_13.add(Dropout(0.5))

model_13.add(Flatten())
model_13.add(Dense(512, activation='relu'))
model_13.add(Dropout(0.5))
model_13.add(Dense(2, activation='softmax'))

## compile a model
model_13.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_13 = model_13.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 13')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model13_hist.pkl', 'wb') as file_name:
    pickle.dump(history_13.history, file_name)

model_13.save('../cnn_model/classifer_1/model13.h5')

print('-----------------------------------')
print('end train model 13')
print('-----------------------------------')

print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 14')
print('-----------------------------------')


## model_14 train
model_14 = Sequential()
model_14.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_14.add(MaxPooling2D(pool_size=(2, 2)))

model_14.add(Dropout(0.5))

model_14.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_14.add(MaxPooling2D(pool_size=(2, 2)))

model_14.add(Dropout(0.5))

model_14.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_14.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_14.add(MaxPooling2D(pool_size=(2, 2)))

model_14.add(Dropout(0.5))

model_14.add(Flatten())
model_14.add(Dense(512, activation='relu'))
model_14.add(Dropout(0.5))
model_14.add(Dense(2, activation='softmax'))

## compile a model
model_14.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_14 = model_14.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 14')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model14_hist.pkl', 'wb') as file_name:
    pickle.dump(history_14.history, file_name)

model_14.save('../cnn_model/classifer_1/model14.h5')

print('-----------------------------------')
print('end train model 14')
print('-----------------------------------')

print('------------------------------------------------------------------------------')



