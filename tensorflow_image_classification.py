import os #for access directory and folder of operating system
import numpy as np
import cv2 #for image conversion means convert into pixle
import matplotlib.pyplot as plt #for show image that are converted into the array of pixel
import random #for shuffle all the traiing_input array

training_data=[] #Traing data will store into that list
category = ["darshak","saurabh","thiren"] 
directory = 'C:/Users/om/Desktop/'
for typee in category:
    path = os.path.join(directory,typee) #show path = "C:/Users/om/Desktop/darshak"
    class_name = category.index(typee) #show index of the catesgory like darshak=0,saurabh=1
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) #convert jpg to pixle of array and also convert into grayscale...........
            new_array = cv2.resize(img_array,(50,50)) #50,50 -> convert size of image into 50*50 pixel
            training_data.append([new_array,class_name]) #append that pixle of array of image(image),index of category(label) into the traiing_data list means [[darshak_image,0],[saurabh_image,1]]..........
        except Exception as e:
            pass
print(len(training_data)) #Total number of input images

random.shuffle(training_data)

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1,50,50,1) # 1-> here image is gray scale so write 1 , if ur image is rgb then write  3.
Y = np.array(Y)
X = X/255.0 #Reduce pixle size
print(Y[20])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1))) # 32->number of nuerons, (3,3)->window size of input, (50,50,1)->image size 50*50 and 1 because image is gray scale....
model.add(Activation('relu')) #Activation function
model.add(MaxPooling2D(pool_size=(2, 2))) #MaxPooling2D is layer which select maximum size window and combine into new 2*2 array so pool_size is (2,2).......

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())# this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64)) # (64)->nuerons in Dense layer
model.add(Activation('relu'))

# model.add(Dropout(0.5))

model.add(Dense(3)) # final 1 output layer
model.add(Activation('softmax'))
# COMPILEself
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])#sparse_categorycal_crossentropy=if more than one class,categorycal_crossentropy=if predict only one class

model.fit(X,Y,epochs=15,batch_size=5,validation_split=0.1) #batch_size=total number of input pass at a time, validation_split=(0.1)10% of input are use for validation......

import cv2
import tensorflow as tf
def convert_image(img):
    ia = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    na = cv2.resize(ia,(50,50))
    return na.reshape(-1,50,50,1)

# t  = np.array(convert_image('C:/Users/om/Desktop/1111.jpg'))
# t = tf.cast(t, tf.float32)
# p = model.predict(t)
# print(p)

import tensorflow as tf
# from tensorflow.contrib import lite
# Save tf.keras model in HDF5 format.
keras_file = "myimage_pred.h5"
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("myimage__pred.tflite", "wb").write(tflite_model)
