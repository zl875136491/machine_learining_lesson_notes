# -*- coding:UTF-8 -*-
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
#构造图（Graph）的结构
#用一个线性方程的例子：y=w*x+t
# 权重
w = tf.Variable(2.0, dtype=tf.float32,name="Weight")
#偏差
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")
#输入
x = tf.placeholder(dtype=tf.float32, name="Input")

#输出命名空间
with tf.name_scope("out_put"):
    y = w * x + b

#定义保存日志的路径
path = "./log"

#创建用于初始化所有变量的操作
#如果定义了变量而未初始化，则会报错
init = tf.global_variables_initializer()

#创建Session
with tf.Session() as sess:
    #初始化变量
    sess.run(init)
    writer = tf.summary.FileWriter(path,sess.graph)
    #为x赋值为3
    result = sess.run(y,{x:3.0})
    print("y = w * x + b 值为：{}".format(result))