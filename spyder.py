#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:27:16 2021

@author: meshal
"""
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
from keras.models import  load_model
#from keras.layers import LSTM , Dense , Embedding , Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, Flatten, Dense , LSTM ,Embedding , Dropout,SpatialDropout1D
from tensorflow.keras import Sequential 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv('Train.csv')

df.isna().sum()
df['Text'].str.len()

df['Text'].str.len().plot(kind='hist')

df['Label'].value_counts()

df['Label'].value_counts().plot(kind='bar')

MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 5000

tokenizer = Tokenizer(num_words=5MAX_NB_WORDS , split=" ")
tokenizer.fit_on_texts(df['Text'].values)

X = tokenizer.texts_to_sequences(df['Text'].values)
X= pad_sequences(X , maxlen=MAX_SEQUENCE_LENGTH)
X[:7]


model = Sequential()
model.add(Embedding(MAX_NB_WORDS,265, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(20, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'])
model.summary()


y = pd.get_dummies(df['Label']).values
[print(df['Label'][i] , y[i]) for i in range (0,40)]
print('Shape of label tensor:', y.shape)

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.3 , random_state=0)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

batch_size = 32
epochs = 2
history = model.fit(X_train , y_train , epochs = epochs , batch_size = batch_size , verbose = 2)


#history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();



plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show();

test_df = pd.read_csv('Test.csv')
print(test_df.head())

new_complaint = ['I am a victim of identity theft and someone stole my identity and personal information to open up a Visa credit card account with Bank of America. The following Bank of America Visa credit card account do not belong to me : XXXX.']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['SOCIAL ISSUES', 'EDUCATION', 'RELATIONSHIPS', 'ECONOMY', 'RELIGION', 'POLITICS', 'LAW/ORDER', 'SOCIAL', 'HEALTH', 'ARTS AND CRAFTS', 'FARMING', 'CULTURE', 'FLOODING', 'WITCHCRAFT', 'MUSIC', 'TRANSPORT', 'WILDLIFE/ENVIRONMENT', 'LOCALCHIEFS', 'SPORTS', 'OPINION/ESSAY']
print(pred, labels[np.argmax(pred)])
























