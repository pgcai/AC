from utils import loaddata
from utils import loss_history
from utils import plot_loss
from keras.utils.vis_utils import plot_model
 
import os
import cv2
import numpy as np
 
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
 
path='/home/fanzy/data/catdog/train/'
size=224
x_train,x_test,y_train,y_test=loaddata.catdogimg(path,size,0.3)
 
model=Sequential()
model.add(Conv2D(64,(3,3),padding='same',input_shape=(size,size,3),activation='tanh'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',activation='tanh'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2),strides=2))
 
model.add(Conv2D(128,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(128,(3,3),padding='same',activation='tanh'))
model.add(MaxPooling2D((2,2),strides=2))
 
model.add(Conv2D(256,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(256,(3,3),padding='same',activation='tanh'))
model.add(MaxPooling2D((2,2),strides=2))
 
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(MaxPooling2D((2,2),strides=2))
 
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(Conv2D(512,(3,3),padding='same',activation='tanh'))
model.add(MaxPooling2D((2,2),strides=2))
 
model.add(Flatten())
model.add(Dense(1024,activation='tanh'))
model.add(Dense(1024,activation='tanh'))
model.add(Dense(1024,activation='tanh'))
model.add(Dense(2,activation='softmax'))
 
plot_model(model,to_file='vgg.png',show_shapes=True,show_layer_names=False)
