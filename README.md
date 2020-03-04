# cnn-classification-dog-vs-cat
基于CNN的图像分类器，使用Kaggle的猫狗图片数据。

## 1 requirement
- python3
- numpy >= 1.14.2
- keras >= 2.1.6
- tensorflow >= 1.6.0
- h5py >= 2.7.0
- python-gflags >= 3.1.2
- opencv-python >= 3.4.0.12

## 2 Description of files
- inputs: 猫狗图片样本数据，[[下载地址]](https://www.kaggle.com/c/dogs-vs-cats/data)，使用keras库中的[ImageDataGenerator](https://keras.io/preprocessing/image/)类读取，需要将每个类的图片放在单独命名的文件夹中存放；
- train.py: 自建的简单CNN，训练后测试集精度约83%；
- pre_train.py: 利用已训练的常用网络(基于[ImageNet](http://www.image-net.org/)数据集训练)，进行迁移学习，测试集精度约95%以上；
- data_helper.py: 数据读取和预处理模块；
- img_cnn.py: 基于TensorFlow的自定义简单卷积神经网络。

## 3 Start training
- ### 训练自定义的小型CNN
    ```shell
    python train.py
    ```
- ### 在VGG16的基础上进行迁移学习
    ```shell
    python pre_train.py
    ```

## 4 Visualizing results in TensorBoard
```shell
tensorboard --logdir /"PATH_OF_CODE"/log/"TIMESTAMP"/summaries/
```

## 5 References
[1]. 猫狗图像数据来源：
https://www.kaggle.com/c/dogs-vs-cats/data

[2]. keras中载入已训练网络的方法：
https://keras.io/applications/

[3]. keras中图像预处理的相关功能介绍：
https://keras.io/preprocessing/image/