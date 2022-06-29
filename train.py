# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:46:29 2022

@author: roy_wu
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

#初始化所需的參數
epochs=10 #訓練的次數
img_rows = None #驗證碼影像檔的高
img_cols = None #驗證碼影像檔的寬
digits_in_img = 6 #驗證碼影像檔中有幾位數
x_list = list() #存所有驗證碼數字影像的array
y_list = list() #存所有驗證碼數字影像檔array代表正確的數字
x_train = list() #存訓練用驗證碼數字影像檔的array
y_train = list() #存訓練用驗證碼數字影像檔array代表的正確數字
x_test = list()   #存測試用驗證碼數字影像檔的array
y_test = list()   #存測試用驗證碼數字影像檔array代表的正確數字

#寫一個將驗證碼6位數獨立切出的funciton
#驗證碼數字影像檔的array會存在x_list
#驗證碼數字影像檔array代表的正確數字會存在y_list

def split_digits_in_img(img_array, x_list, y_list):
    for i in range(digits_in_img):
        step = img_cols // digits_in_img
        x_list.append(img_array[:, i * step:(i + 1) * step] / 255)
        y_list.append(img_filename[i])
        
        
img_filenames = os.listdir('training')
 
for img_filename in img_filenames:
    if '.png' not in img_filename:
        continue
    img = load_img('training/{0}'.format(img_filename), color_mode='grayscale')
    img_array = img_to_array(img)
    img_rows, img_cols, _ = img_array.shape
    split_digits_in_img(img_array, x_list, y_list)