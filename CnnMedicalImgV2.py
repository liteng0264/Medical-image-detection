#coding=utf-8
import cv2  #pip install opencv_python-4.2.0-cp37-cp37m-win_amd64.whl
import glob #查找符合特定规则的文件路径名
import random
import numpy as np
from sklearn.model_selection import train_test_split #用来划分训练集和验证集
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics,losses

#如果安装的是CPU版本（pip install tensorflow）
#在代码中加入如下代码，忽略警告
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 7
np.random.seed(seed)
 
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
    #if fileindex > 500:
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
X_train, X_test, y_train, y_test = train_test_split(TrainX_rgb, TrainY_rgb, test_size=0.10, random_state=seed)

#db = tf.data.Dataset.from_tensor_slices((X_train,y_train))
#db = db.batch(100)

train_db = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_db = train_db.shuffle(1000).batch(100)

test_db = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_db = test_db.batch(100)



# 创建网络模型并装配模型
network = Sequential([
    layers.Conv2D(6, kernel_size=5, strides=1, padding='SAME', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Conv2D(16, kernel_size=5, strides=1, padding='SAME', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation=tf.nn.sigmoid)
])
network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=losses.binary_crossentropy,
                metrics=['accuracy'])


# 显示网络结构信息
network.build(input_shape=[None, 224, 224, 3])
network.summary()


# 设置回调功能
filepath = 'C:\\MedicalImgs\\my_model.h5' # 保存模型地址
saved_model = tf.keras.callbacks.ModelCheckpoint(filepath, verbose = 1) # 回调保存模型功能
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'log') # 回调可视化数据功能


# 执行训练与验证
history = network.fit(train_db, epochs = 2, validation_data = test_db, validation_freq = 1,
                      callbacks = [saved_model, tensorboard])

# 显示训练与验证相关数据统计
history.history


# 加载训练好的模型并进行测试
del network
print("Loading model..")
network = tf.keras.models.load_model('C:\\MedicalImgs\\my_model.h5')
print("Complete load model.")
loss, acc = network.evaluate(test_db)
print("Test accuracy: %g%%" % (acc * 100)) # 这里只测试了一个batch的准确度