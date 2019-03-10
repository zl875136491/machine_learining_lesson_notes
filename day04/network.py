# -*- coding:UTF-8 -*-
# 深度卷积的生成对抗网络
import tensorflow as tf


# 超参:学习率，步长，单批图片数量，迭代次数
LEARNING_RATE = 0.0002
BETA_1 = 0.5
BATCH_SIZE = 128
EPOCHS = 6


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
