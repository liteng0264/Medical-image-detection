#coding=utf-8
import cv2  #pip install opencv_python-4.2.0-cp37-cp37m-win_amd64.whl
import glob #查找符合特定规则的文件路径名
import random
import numpy as np
import os

from sklearn.model_selection import train_test_split #用来划分训练集和验证集
import tensorflow as tf #conda install tensorflow
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(2345)


#加载数据：
seed = 7
np.random.seed(seed)
 
#-------------------------------------GPU
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#----------------------------model definition
def Model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((224, 224,3), input_shape=(224, 224,3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model



#------------------------------------------------------------------------------dataset
#opencv不支持中文路径
ImagesFilePath = r'F:/MedicalImgs'
filenames = glob.glob(ImagesFilePath + "/*.jpg")
fliescount = len(filenames)
print("files count: %d" % fliescount)
totalfiles = fliescount #图片的总数量
TrainX_rgb = np.zeros([totalfiles ,224,224,3]) #图片
TrainY_rgb = np.zeros([totalfiles]) #值
fileindex = 0 #当前的文件索引
for filename in filenames:
    #if fileindex > 100:
    #    break #用于测试，只读入几张
    imgsdata = cv2.imread(filename)
    TrainX_rgb[fileindex] = imgsdata
    basefilename = os.path.basename(filename) 
    if basefilename[0:2] == 'CD':
        TrainY_rgb[fileindex] = 1 # CD
    else: 
        if basefilename[0:2] == 'UC':
            TrainY_rgb[fileindex] = 0 # UC
        else:
            assert(1)

    #next file
    fileindex = fileindex + 1

#归一化
TrainX_rgb = TrainX_rgb.astype("float32")
TrainX_rgb /= 255
TrainX_rgb = TrainX_rgb.reshape(TrainX_rgb.shape[0],224,224,3)
TrainY_rgb = TrainY_rgb.astype("float32")
TrainY_rgb = TrainY_rgb.reshape((TrainY_rgb.shape[0], 1))

# split into 67% for train and 33% for test
x_train, x_test, y_train, y_test = train_test_split(TrainX_rgb, TrainY_rgb, test_size=0.20, random_state=seed)

#------------------------------------------------------------------------------model
#compile
model = Model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train,
          y_train,
          batch_size=128,
          epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
