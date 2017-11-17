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

## add shuffle
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
print('train model 1')
print('-----------------------------------')

# SUMMARY
#
# Model 1 - 8 : variation of classifer 2 (Ying2017)
# Model 9 - 11: variation of classifer 1 from Kaggle

## model_1 train
model_1 = Sequential()
model_1.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))

#model_1.add(Dropout(0.5))

model_1.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))

#model_1.add(Dropout(0.5))

model_1.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_1.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))

#model_1.add(Dropout(0.5))

model_1.add(Flatten())
model_1.add(Dense(512, activation='relu'))
#model_1.add(Dropout(0.5))
model_1.add(Dense(2, activation='softmax'))

## compile a model
model_1.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_1 = model_1.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 1')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model1_hist.pkl', 'wb') as file_name:
    pickle.dump(history_1.history, file_name)

model_1.save('../cnn_model/classifer_1/model1.h5')

print('-----------------------------------')
print('end train model 1')
print('-----------------------------------')

print('----------------------------------------------------------------')


## model_2 train
model_2 = Sequential()
model_2.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.2))

model_2.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.2))

model_2.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_2.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_2.add(MaxPooling2D(pool_size=(2, 2)))

model_2.add(Dropout(0.2))

model_2.add(Flatten())
model_2.add(Dense(512, activation='relu'))
model_2.add(Dropout(0.2))
model_2.add(Dense(2, activation='softmax'))

## compile a model
model_2.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_2 = model_2.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 2')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model2_hist.pkl', 'wb') as file_name:
    pickle.dump(history_2.history, file_name)

model_1.save('../cnn_model/classifer_1/model2.h5')

print('-----------------------------------')
print('end train model 2')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 3')
print('-----------------------------------')


## model_3 train
model_3 = Sequential()
model_3.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Dropout(0.2))

model_3.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Dropout(0.2))

model_3.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_3.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_3.add(MaxPooling2D(pool_size=(2, 2)))

model_3.add(Dropout(0.2))

model_3.add(Flatten())
model_3.add(Dense(512, activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(2, activation='softmax'))

## compile a model
model_3.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_3 = model_3.fit(x, y, batch_size=128, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 3')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model3_hist.pkl', 'wb') as file_name:
    pickle.dump(history_3.history, file_name)

model_1.save('../cnn_model/classifer_1/model3.h5')

print('-----------------------------------')
print('end train model 3')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 4')
print('-----------------------------------')


## model_4 train
model_4 = Sequential()
model_4.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_4.add(MaxPooling2D(pool_size=(2, 2)))

model_4.add(Dropout(0.2))

model_4.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_4.add(MaxPooling2D(pool_size=(2, 2)))

model_4.add(Dropout(0.2))

model_4.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_4.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_4.add(MaxPooling2D(pool_size=(2, 2)))

model_4.add(Dropout(0.2))

model_4.add(Flatten())
model_4.add(Dense(512, activation='relu'))
model_4.add(Dropout(0.2))
model_4.add(Dense(2, activation='softmax'))

## compile a model
model_4.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_4 = model_4.fit(x, y, batch_size=32, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 4')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model4_hist.pkl', 'wb') as file_name:
    pickle.dump(history_4.history, file_name)

model_4.save('../cnn_model/classifer_1/model4.h5')

print('-----------------------------------')
print('end train model 4')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 5')
print('-----------------------------------')


## model_5 train
model_5 = Sequential()
model_5.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_5.add(MaxPooling2D(pool_size=(2, 2)))

model_5.add(Dropout(0.2))

model_5.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_5.add(MaxPooling2D(pool_size=(2, 2)))

model_5.add(Dropout(0.2))

model_5.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_5.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_5.add(MaxPooling2D(pool_size=(2, 2)))

model_5.add(Dropout(0.2))

model_5.add(Flatten())
model_5.add(Dense(512, activation='relu'))
model_5.add(Dropout(0.2))
model_5.add(Dense(2, activation='softmax'))

## compile a model
model_5.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_5 = model_5.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 5')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model5_hist.pkl', 'wb') as file_name:
    pickle.dump(history_5.history, file_name)

model_5.save('../cnn_model/classifer_1/model5.h5')

print('-----------------------------------')
print('end train model 5')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 6')
print('-----------------------------------')


## model_6 train
model_6 = Sequential()
model_6.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_6.add(MaxPooling2D(pool_size=(2, 2)))

model_6.add(Dropout(0.5))

model_6.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_6.add(MaxPooling2D(pool_size=(2, 2)))

model_6.add(Dropout(0.5))

model_6.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_6.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_6.add(MaxPooling2D(pool_size=(2, 2)))

model_6.add(Dropout(0.5))

model_6.add(Flatten())
model_6.add(Dense(512, activation='relu'))
model_6.add(Dropout(0.5))
model_6.add(Dense(2, activation='softmax'))

## compile a model
model_6.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_6 = model_6.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 6')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model6_hist.pkl', 'wb') as file_name:
    pickle.dump(history_6.history, file_name)

model_6.save('../cnn_model/classifer_1/model6.h5')

print('-----------------------------------')
print('end train model 6')
print('-----------------------------------')

print('----------------------------------------------------------------')

print('-----------------------------------')
print('train model 7')
print('-----------------------------------')


## model_7 train
model_7 = Sequential()
model_7.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_7.add(MaxPooling2D(pool_size=(2, 2)))

model_7.add(Dropout(0.5))

model_7.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_7.add(MaxPooling2D(pool_size=(2, 2)))

model_7.add(Dropout(0.5))

model_7.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_7.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_7.add(MaxPooling2D(pool_size=(2, 2)))

model_7.add(Dropout(0.5))

model_7.add(Flatten())
model_7.add(Dense(512, activation='relu'))
model_7.add(Dropout(0.5))
model_7.add(Dense(2, activation='softmax'))

## compile a model
model_7.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_7 = model_7.fit(x, y, batch_size=64, epochs=300, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 7')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model7_hist.pkl', 'wb') as file_name:
    pickle.dump(history_7.history, file_name)

model_7.save('../cnn_model/classifer_1/model7.h5')

print('-----------------------------------')
print('end train model 7')
print('-----------------------------------')

print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 8')
print('-----------------------------------')


## model_8 train
model_8 = Sequential()
model_8.add(Conv2D(64, (5, 5), padding="valid", input_shape=(80, 80, 3), activation='relu'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))

model_8.add(Dropout(0.5))

model_8.add(Conv2D(128, (5, 5), padding="valid", activation='relu'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))

model_8.add(Dropout(0.5))

model_8.add(Conv2D(256, (3, 3), padding="valid", activation='relu'))
model_8.add(Conv2D(384, (3, 3), padding="valid", activation='relu'))
model_8.add(MaxPooling2D(pool_size=(2, 2)))

model_8.add(Dropout(0.5))

model_8.add(Flatten())
model_8.add(Dense(512, activation='relu'))
model_8.add(Dropout(0.5))
model_8.add(Dense(2, activation='softmax'))

## compile a model
model_8.compile(loss='categorical_crossentropy', optimizer= Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

## train a model

train_start = time.time()

history_8 = model_8.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 8')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model8_hist.pkl', 'wb') as file_name:
    pickle.dump(history_8.history, file_name)

model_8.save('../cnn_model/classifer_1/model8.h5')

print('-----------------------------------')
print('end train model 8')
print('-----------------------------------')

print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 9')
print('-----------------------------------')

##########################
## Classifer CNN Kaggle ##
##########################

## Model 9 is Classifer CNN

## model_9 train
model_9 = Sequential()
model_9.add(Conv2D(32, (3, 3), padding="same", input_shape=(80, 80, 3), activation='relu'))
model_9.add(Conv2D(32, (3, 3), activation='relu'))
model_9.add(MaxPooling2D(pool_size=(2, 2)))

model_9.add(Dropout(0.2))

model_9.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model_9.add(Conv2D(64, (3, 3), activation='relu'))
model_9.add(MaxPooling2D(pool_size=(2, 2)))

model_9.add(Flatten())
model_9.add(Dense(512, activation='relu'))
model_9.add(Dropout(0.2))
model_9.add(Dense(2, activation='softmax'))

## compile a model
model_9.compile(loss='categorical_crossentropy', optimizer= SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_9 = model_9.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 9')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model9_hist.pkl', 'wb') as file_name:
    pickle.dump(history_9.history, file_name)

model_9.save('../cnn_model/classifer_1/model9.h5')

print('-----------------------------------')
print('end train model 9')
print('-----------------------------------')

print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 10')
print('-----------------------------------')


##########################
## Classifer CNN Kaggle ##
##########################

## Model 10 is Classifer CNN

## model_10 train
model_10 = Sequential()
model_10.add(Conv2D(32, (3, 3), padding="same", input_shape=(80, 80, 3), activation='relu'))
model_10.add(Conv2D(32, (3, 3), activation='relu'))
model_10.add(MaxPooling2D(pool_size=(2, 2)))

model_10.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model_10.add(Conv2D(64, (3, 3), activation='relu'))
model_10.add(MaxPooling2D(pool_size=(2, 2)))

model_10.add(Dropout(0.2))

model_10.add(Flatten())
model_10.add(Dense(512, activation='relu'))
model_10.add(Dropout(0.2))
model_10.add(Dense(2, activation='softmax'))

## compile a model
model_10.compile(loss='categorical_crossentropy', optimizer= SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_10 = model_10.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 10')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model10_hist.pkl', 'wb') as file_name:
    pickle.dump(history_10.history, file_name)

model_10.save('../cnn_model/classifer_1/model10.h5')

print('-----------------------------------')
print('end train model 10')
print('-----------------------------------')


print('------------------------------------------------------------------------------')

print('-----------------------------------')
print('train model 10')
print('-----------------------------------')

##########################
## Classifer CNN Kaggle ##
##########################

## Model 11 is Classifer CNN

## model_11 train
model_11 = Sequential()
model_11.add(Conv2D(32, (3, 3), padding="same", input_shape=(80, 80, 3), activation='relu'))
model_11.add(Conv2D(32, (3, 3), activation='relu'))
model_11.add(MaxPooling2D(pool_size=(2, 2)))

model_11.add(Dropout(0.2))

model_11.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model_11.add(Conv2D(64, (3, 3), activation='relu'))
model_11.add(MaxPooling2D(pool_size=(2, 2)))

model_11.add(Flatten())
model_11.add(Dense(512, activation='relu'))
model_11.add(Dropout(0.2))
model_11.add(Dense(2, activation='softmax'))

## compile a model
model_11.compile(loss='categorical_crossentropy', optimizer= SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=['accuracy'])

## train a model

train_start = time.time()

history_11 = model_11.fit(x, y, batch_size=64, epochs=150, validation_split=0.2)

train_end = time.time()

train_time = train_end - train_start
print ('CNN model train time = {}'.format(train_time))

print('-----------------------------------')
print('save model 11')
print('-----------------------------------')

with open('../cnn_model/classifer_1/model11_hist.pkl', 'wb') as file_name:
    pickle.dump(history_11.history, file_name)

model_11.save('../cnn_model/classifer_1/model11.h5')

print('-----------------------------------')
print('end train model 11')
print('-----------------------------------')

