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
        #if epoch % 10 == 9:
        g.save_weights('generator_weight.h5', True)
        d.save_weights('discriminator_weight.h5', True)


if __name__ == "__main__":
    train()
