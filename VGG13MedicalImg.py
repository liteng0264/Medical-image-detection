#coding=utf-8
import cv2  #pip install opencv_python-4.2.0-cp37-cp37m-win_amd64.whl
import glob #查找符合特定规则的文件路径名
import random
import numpy as np
import os

from sklearn.model_selection import train_test_split #用来划分训练集和验证集
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(2345)


#加载数据：
seed = 7
np.random.seed(seed)
 
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#opencv不支持中文路径
ImagesFilePath = r'C:/MedicalImgs'
filenames = glob.glob(ImagesFilePath + "/*.jpg")
fliescount = len(filenames)
print("files count: %d" % fliescount)
totalfiles = fliescount #视频的总数量
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
x_train, x_test, y_train, y_test = train_test_split(TrainX_rgb, TrainY_rgb, test_size=0.10, random_state=seed)
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(1000).batch(64)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(64)

#网络

conv_layers=[
    layers.Conv2D(64,kernel_size=[3,3],padding="valid",activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding="valid"),

    layers.Conv2D(128, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid"),

    layers.Conv2D(256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid"),

    layers.Conv2D(512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=3, padding="valid"),

    layers.Conv2D(512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="valid", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid"),
]

conv_net=Sequential(conv_layers)
#conv_net.build(input_shape=[None, 224, 224, 3])
#x=tf.random.normal([64,224,224,3])
#out=conv_net(x)
#print(out.shape)
fc_net=Sequential([
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(1,activation=tf.nn.sigmoid),
])
conv_net.build(input_shape=[None, 224, 224, 3])
fc_net.build(input_shape=[None,512])
optimizer=optimizers.Adam(lr=1e-4)

#计算loss:
variables=conv_net.trainable_variables+fc_net.trainable_variables
for epoch in range(50):
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out=conv_net(x)
            out=tf.reshape(out,[-1,512])
            logits=fc_net(out)
            #y_onehot=tf.one_hot(y,depth=10)
            loss=tf.losses.binary_crossentropy(y,logits)
            loss=tf.reduce_mean(loss)
        grads=tape.gradient(loss,variables)
        optimizer.apply_gradients(zip(grads,variables))
        if step%100==0:
            print(epoch,step,'loss',float(loss))

#测试
total_num=0
total_correct=0
for x,y in test_db:
    out=conv_net(x)
    out=tf.reshape(out,[-1,512])
    logits=fc_net(out)
    prob=tf.nn.softmax(logits,axis=1)
    pred=tf.argmax(prob,axis=1)
    pred=tf.cast(pred,dtype=tf.int32)
    correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
    correct=tf.reduce_sum(correct)
    total_num+=x.shape[0]
    total_correct+=int(correct)
acc=total_correct/total_num
print(epoch,'acc:',acc)
