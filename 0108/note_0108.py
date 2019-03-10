---------------------------------------CODE6--------------------------------------------------


# -*- coding:UTF-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

# one_hot 独热码的编码格式
mnist = input_data.read_data_sets("./mnist_data", one_hot=True)


#None表示张量的第一个维度可以是任何长度
#除以255是为了做 归一化（Normalization），把灰度从[0, 255]，变成[0,1]，区间
#归一化的目的：可以让之后的优化器（optimizer）更快更好地找到误差最小值
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) /255

#输出：10个数字的标签
out_put_y = tf.placeholder(tf.int32, [None, 10])

#改变形状之后的输入
#-1表示自动推导维度的大小
#让算法根据其他维度的值和总的元素大小来推导出-1处的维度应该为多少
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])


#从Test数据集中选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]


#构建卷积神经网络
#第一层卷积   (28, 28, 1)  ->  (28, 28, 32)
conv1 = tf.layers.conv2d(

    inputs=input_x_images,      #形状[28, 28, 1]
    filters=32,                #32个过滤器
    kernel_size=[5, 5],         #过滤器在二维的大小（5*5）
    strides=1,                  #步长是1
    padding='same',             #same表示输出大小不变，因此需要在外围补零
    activation=tf.nn.relu
)


#第一层池化（亚采样）
#形状      (28, 28, 32) ->  (14, 14, 32)
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,               #形状[28, 28, 32]
    pool_size=[2, 2],            #过滤器在二维的大小是（2*2）
    strides=2,                  #步长是2
)


#第二层卷积
#形状      (14, 14, 32) ->  (14, 14, 64)
conv2 = tf.layers.conv2d(
    inputs=pool1,               #形状(14, 14, 32)
    filters=64,                 #64个过滤器，输出的深度（depth）：64
    kernel_size=[5, 5],
    strides=1,                  #步长：1
    padding='same',
    activation=tf.nn.relu
    )


#第二层池化（亚采样）
#形状     (14, 14, 64) ->   (7, 7, 64)
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2,
)


#平坦化降维（flat）
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


#全连接层
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)


#Dropout：丢弃 50%
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

#10个神经元的全连接层
#形状(1, 1, 10)
logits = tf.layers.dense(inputs=dropout, units=10)

#logits = tf.layers.dense(inputs=flat, units=10)
#计算误差
#先用softmax计算百分比概率，再用cross_entropy（交叉熵）来计算独热码和百分比概率之间的误差
loss = tf.losses.softmax_cross_entropy(onehot_labels=out_put_y, logits=logits)


#用优化器来最小化误差
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


#计算预测值和实际标签的匹配程度
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(out_put_y, axis=1),
    predictions=tf.argmax(logits, axis=1),
)[1]


#创建会话
sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

#训练5000步
for i in range(5000):
    batch = mnist.train.next_batch(50)          #从train数据集里取下一个共50个样本
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], out_put_y:  batch[1]})     #损失，精度
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x:test_x, out_put_y:test_y})
        print("The {} step 's loss is {:.4f}, op is {:.2f}".format(i, train_loss, test_accuracy))


#打印20个预测值和真实值
test_output = sess.run(logits, {input_x:test_x[:20]})
inferred_y = np.argmax(test_output, 1)
print(inferred_y, "inferred numbers")
print(np.argmax(test_y[:20], 1), "real number")

---------------------------------------CODE6----------------------------------------------------




---------------------------------------RESULT---------------------------------------------------
'''
The 0 step 's loss is 2.3158, op is 0.09
The 100 step 's loss is 0.1883, op is 0.50
The 200 step 's loss is 0.2018, op is 0.65
The 300 step 's loss is 0.1185, op is 0.73
The 400 step 's loss is 0.0178, op is 0.77
The 500 step 's loss is 0.0822, op is 0.81
The 600 step 's loss is 0.1691, op is 0.83
The 700 step 's loss is 0.0239, op is 0.85
The 800 step 's loss is 0.0383, op is 0.86
The 900 step 's loss is 0.1333, op is 0.87
The 1000 step 's loss is 0.0043, op is 0.88
The 1100 step 's loss is 0.0205, op is 0.89
The 1200 step 's loss is 0.0728, op is 0.90
The 1300 step 's loss is 0.0323, op is 0.90
The 1400 step 's loss is 0.0052, op is 0.91
The 1500 step 's loss is 0.2194, op is 0.91
The 1600 step 's loss is 0.0080, op is 0.92
The 1700 step 's loss is 0.0673, op is 0.92
The 1800 step 's loss is 0.0251, op is 0.92
The 1900 step 's loss is 0.0285, op is 0.93
The 2000 step 's loss is 0.1103, op is 0.93
The 2100 step 's loss is 0.0939, op is 0.93
The 2200 step 's loss is 0.0394, op is 0.93
The 2300 step 's loss is 0.0199, op is 0.94
The 2400 step 's loss is 0.0134, op is 0.94
The 2500 step 's loss is 0.0942, op is 0.94
The 2600 step 's loss is 0.0613, op is 0.94
The 2700 step 's loss is 0.0493, op is 0.94
The 2800 step 's loss is 0.0520, op is 0.94
The 2900 step 's loss is 0.0950, op is 0.95
The 3000 step 's loss is 0.0201, op is 0.95
The 3100 step 's loss is 0.0042, op is 0.95
The 3200 step 's loss is 0.0230, op is 0.95
The 3300 step 's loss is 0.0012, op is 0.95
The 3400 step 's loss is 0.0180, op is 0.95
The 3500 step 's loss is 0.0384, op is 0.95
The 3600 step 's loss is 0.0005, op is 0.95
The 3700 step 's loss is 0.0046, op is 0.95
The 3800 step 's loss is 0.1018, op is 0.95
The 3900 step 's loss is 0.0796, op is 0.95
The 4000 step 's loss is 0.0042, op is 0.96
The 4100 step 's loss is 0.0109, op is 0.96
The 4200 step 's loss is 0.0323, op is 0.96
The 4300 step 's loss is 0.0381, op is 0.96
The 4400 step 's loss is 0.0126, op is 0.96
The 4500 step 's loss is 0.0303, op is 0.96
The 4600 step 's loss is 0.0070, op is 0.96
The 4700 step 's loss is 0.0045, op is 0.96
The 4800 step 's loss is 0.0034, op is 0.96
The 4900 step 's loss is 0.0005, op is 0.96
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] inferred numbers
[7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4] real number
'''
-------------------------------------RESULT---------------------------------------------------




