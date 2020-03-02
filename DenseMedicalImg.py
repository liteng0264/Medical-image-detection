#coding=utf-8
import cv2  #pip install opencv_python-4.2.0-cp37-cp37m-win_amd64.whl
import glob #查找符合特定规则的文件路径名
import random
import numpy as np
from sklearn.model_selection import train_test_split #用来划分训练集和验证集
import  tensorflow as tf

#如果安装的是CPU版本（pip install tensorflow）
#在代码中加入如下代码，忽略警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

seed = 7
np.random.seed(seed)
 
#opencv不支持中文路径
ImagesFilePath = r'F:/MedicalImgs'
filenames = glob.glob(ImagesFilePath + "/*.jpg")
fliescount = len(filenames)
print("files count: %d" % fliescount)
totalfiles = fliescount #视频的总数量
TrainX_rgb = np.zeros([totalfiles ,224,224]) #视频数量,有4个modality
TrainY_rgb = np.zeros([totalfiles ]) #视频数量

fileindex = 0 #当前的文件索引
for filename in filenames:
    #if fileindex>500:
    #    break  #用于测试，只读入几张
    imgsdata = cv2.imread(filename)
    #print(imgsdata['img_rgb'].shape)
    frame1_gray = cv2.cvtColor(imgsdata, cv2.COLOR_BGR2GRAY)   #转换了灰度化
    frame1 = cv2.resize(frame1_gray, (224, 224), interpolation=cv2.INTER_CUBIC)
    TrainX_rgb[fileindex ] = frame1
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

TrainX_rgb = TrainX_rgb.astype("float32")
TrainX_rgb /= 255
#TrainX_rgb = TrainX_rgb.reshape(TrainX_rgb.shape[0],224,224,1)
TrainX_rgb = TrainX_rgb.reshape(TrainX_rgb.shape[0],224*224)
#TrainY_rgb = to_categorical(TrainY_rgb,2)
TrainY_rgb = TrainY_rgb.astype("float32")
TrainY_rgb = TrainY_rgb.reshape((TrainY_rgb.shape[0], 1))

# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(TrainX_rgb, TrainY_rgb, test_size=0.10, random_state=seed)

db = tf.data.Dataset.from_tensor_slices((X_train,y_train))
db = db.batch(32).repeat(10)

# model definition
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(1, activation='sigmoid')])
network.build(input_shape=(None, 224*224))
network.summary()

optimizer = optimizers.SGD(lr=0.001)
acc_meter = metrics.Accuracy()

for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        out = network(x)
        loss = tf.square(out-y)
        loss = tf.reduce_sum(loss)

    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()