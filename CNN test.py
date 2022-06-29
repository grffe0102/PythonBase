# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:45:06 2022

@author: roy_wu
"""
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train /255
x_test = x_test/255

print("x_train.shape:",x_train.shape)
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))
print("x_train.reshape:",x_train.shape)
#第一層 Sequential序列
CNN = keras.Sequential(name = 'CNN')
CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
CNN.add(layers.MaxPooling2D((2,2)))
#第二層
CNN.add(layers.Conv2D(64,(3,3),activation='relu'))
CNN.add(layers.MaxPooling2D((2,2)))
#平坦化
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
"""

import tensorflow as tf
import os
import numpy as np
import glob
import cv2
from tensorflow import keras
from tensorflow.keras import layers
 
#---------------------------------第一步 读取图像-----------------------------------
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    fpath = []
    #print(cate)
    for idx, folder in enumerate(cate):
        # 遍历整个目录判断每个文件是不是符合
        for im in glob.glob(folder + '/*.jpg'):
            #print('reading the images:%s' % (im))
            img = cv2.imread(im)             #调用opencv库读取像素点
            img = cv2.resize(img, (100, 100))  #图像像素大小一致
            imgs.append(img)                 #图像数据
            labels.append(idx)               #图像类标
            fpath.append(path+im)            #图像路径名
            #print(path+im, idx)
            
    return np.asarray(fpath, np.string_), np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

#定義路徑
path = "C:\\Users\\roy_wu\\Desktop\\Tensorflow\\"
# 讀取图像
fpaths, data, label = read_img(path)  # (1000, 256, 256, 3)
print("data.shape:",data.shape)  
# 計算有多少類圖片
num_classes = len(set(label))
print("num_classes:",num_classes)


# 生成等差数列隨機調整影像順序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
fpaths = fpaths[arr]

# 拆分訓練集和測試集 80%訓練集 20%測試集
ratio = 0.8
s = np.int64(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
fpaths_train = fpaths[:s] 
x_test = data[s:]
y_test = label[s:]
fpaths_test = fpaths[s:] 
print("x_train len:",len(x_train),
      "y_train len:",len(y_train),
      "x_val len:",len(x_test),
      "y_val len:",len(y_test)) 
print(y_test)


#-------------------------------------------------
x_train = x_train /255
x_test = x_test/255


#第一層 Sequential序列
CNN = keras.Sequential(name = 'CNN')
CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (100,100,3))) 
#CNN.add(layers.MaxPooling2D((2,2)))
#第二層
CNN.add(layers.Conv2D(64,(3,3),activation='relu'))
#CNN.add(layers.MaxPooling2D((2,2)))
#平坦化
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

#print(CNN.summary())

#save model 
CNN.save('cnn_model.h5')

#預測
predict = CNN.predict(x_test)
predict = np.argmax(predict,axis=1)
print(np.mean(predict))
