mmlab.ie.cuhk.edu.hk
kaggle


什么是GAN？
	Generative Adversarial Network：生成对抗网络
	让两个神经网络相互博弈的方式进行学习
	一个生成网络，一个判别网络

什么是DCGAN
	深度卷积生成对抗网络
	生成模型和判别模型都运用了深度卷积网络的生成对抗网络
	生成：反卷积
	判别：卷积



---------------------------------------CODE7----------------------------------------------------
---------------------------------------network.py-----------------------------------------------
# -*- coding:UTF-8 -*-
# 深度卷积的生成对抗网络
import tensorflow as tf


# 超参:学习率，步长，单批图片数量，迭代次数
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BATCH_SIZE = 128
EPOCHS = 100


# 定义判别器模型:利用高级框架Keras搭建的深度卷积
def discriminator_mode():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64,                         # 过滤器，输出的深度
        (5, 5),                     # 过滤器在二维图像上的大小
        padding='same',             # 输出大小不变
        input_shape=(64, 64, 3)     # 输入形状(64, 64, 3)，3表示通道数，即RGB三色
    ))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 添加池化层     [64,64,64] -> [32,32,64] -> [32,32,128]
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 添加池化层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 添加池化层     [] -> []
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 平坦化
    model.add(tf.keras.layers.Flatten())
    # 1024个神经元的全连接层
    model.add(tf.keras.layers.Dense(1024))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 1个神经元的全连接层
    model.add(tf.keras.layers.Dense(1))
    # 添加Sigmoid激活层
    model.add(tf.keras.layers.Activation("sigmoid"))

    return model


# 定义生成器模型
# 从随机数来生成图片
def generator_model():
    model = tf.keras.models.Sequential()
    # 输入的维度是100 输出的神经元的神经元是1024的全连接层
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 128 * 8 * 8个神经元的全连接层
    model.add(tf.keras.layers.Dense(128 * 8 * 8))
    # 对所有全连接层进行批标准化
    model.add(tf.keras.layers.BatchNormalization())
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 改变形状
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 8 * 8, )))
    # 16 * 16 像素
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 32 * 32 像素
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))
    # 64 * 64 像素
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding='same'))        # [64, 64, 3]
    # 添加Tanh激活层
    model.add(tf.keras.layers.Activation("tanh"))

    return model


# 绑定生成-判别模型
# 构造一个 Sequential对象， 包含一个生成器和判别器
def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    # 初始时 判别器不可被训练
    discriminator.trainable = False
    model.add(discriminator)

    return model



---------------------------------------CODE7----------------------------------------------------




---------------------------------------CODE8----------------------------------------------------
---------------------------------------train.py-------------------------------------------------
# -*- coding:UTF-8 -*-
# 训练DCGAN

from scipy import misc
import os
import glob
import numpy as np
from network import *


def train():
    if not os.path.exists("images"):
        raise Exception("Cant find images")
    data = []
    for image in glob.glob("images/*"):
        # imread 是利用PIL来读取图片数据
        image_data = misc.imread(image)
        data.append(image_data)
    input_data = np.array(data)
    # 将输入标准化成[-1, 1]的取值，这个也是tanh激活函数的输出范围
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5
    # 构造生成器（g）和判别器（d）
    g = generator_model()
    d = discriminator_mode()
    # 构建生成器和判别器组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)
    # 优化器选择 Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # 配置生成器和判别器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    # 开始训练
    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            # 连续性均匀分布的随机数据（噪声）
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            # 生成器 生成的图片数据
            generator_images = g.predict(random_data, verbose=0)  # verbose =2
            # 将输入的数据
            input_batch = np.concatenate((input_batch, generator_images))
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            # 训练判别器：识别生成器图片是否合格
            d_loss = d.train_on_batch(input_batch, output_batch)

            # 当训练生活器时 禁止判别器训练
            d.trainable = False

            # 重新生成随机数据
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            # 训练生成器，并通过‘不可训练’的判别器判别合格与否
            g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)
            # 恢复 判别器 可被训练
            d.trainable = True
            # 打印损失
            print("Epoch {},第{}步,G-loss is {:.3f},D-loss is {:.3f}".format(epoch, index, g_loss, d_loss))
        # 保存 生成器 和判别器 的参数
        if epoch % 10 == 9:
            g.save_weights("generator_weight", True)
            d.save_weights("discriminator_weight", True)




if __name__ =="__main__":
    train()

---------------------------------------CODE8----------------------------------------------------