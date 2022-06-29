# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:42:43 2022

@author: roy_wu
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model


(x_train,y_train),(x_test ,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

x_train = x_train.reshape((60000, 28, 28,1))
x_test = x_test.reshape((10000,28,28,1))


#第一層 Sequential序列
CNN = keras.Sequential(name = 'CNN')
CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
CNN.add(layers.MaxPooling2D((2,2))) 
CNN.add(layers.Conv2D(64,(3,3),activation='relu'))
CNN.add(layers.MaxPooling2D((2,2)))

CNN.add(layers.Flatten())
CNN.add(layers.Dense(128,activation='relu'))
CNN.add(layers.Dense(64,activation='relu'))
CNN.add(layers.Dense(10,activation='softmax'))

keras.utils.plot_model(CNN,show_shapes=True, to_file='model.png')

#compile
CNN.compile(optimizer = 'Adam',
             loss = keras.losses.sparse_categorical_crossentropy,
             metrics = ['accuracy'])

CNN.fit(x_train,y_train,epochs=1)




