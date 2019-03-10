
pycharm 绘制练习
-----------------------------------CODE3------------------------------------------------
# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2,2,100)

y1 = 3 * x + 4
y2 = x ** 3

plt.plot(x,y1)
plt.plot(x,y2)
plt.show()
-----------------------------------CODE3------------------------------------------------




梯度下降
-----------------------------------CODE4------------------------------------------------
# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#构建数据
points_num = 100
vectors = []

#使用numpy的正态随机分布生成100个随机的点
#生成的点（x，y）对应的线性方程 y = w（权重） * x + b（偏差）
for i in range(points_num):
    x1 = np.random.normal(0.0,0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0,0.04)
    vectors.append([x1,y1])

#真实点x的坐标，y的坐标
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

#可视化
#红色星型点：
plt.plot(x_data, y_data, 'r*' , label = "Original data")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()


#构建回归模型
#初始化 weight bias
w = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

#构建代价函数 loss function / cost function
#对张量的所有维度计算(y - y_data)^2 之和除以n
loss = tf.reduce_mean(tf.square(y - y_data))

#用梯度下降模型来最小化损失
#参数0.5为学习率，学习率为步长（越小越好）
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#创建会话
sess = tf.Session()
#初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)

#训练20步
for step in range(20):
    #优化每一步，打印每一步 损失|权重|偏差
    sess.run(train)
    print("the {} step 's weight = {},bias = {}".format(step+1,sess.run(loss),sess.run(w),sess.run(b)))

#绘制图像：绘制所有点，并绘制最佳拟合直线
plt.plot(x_data,y_data,"r*",label = "Original data")
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(w) * x_data + sess.run(b),label = "Fitted data")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
sess.close()


-----------------------------------CODE4------------------------------------------------


TensorFlow激活函数
通过激活函数进行数据分类（非线性的）
e·g： playground演示即为数据分类

-----------------------------------CODE5------------------------------------------------

# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
#创建数据
#（-7，7）内等间隔选取180个点
x = np.linspace(-7, 7, 180)


#激活函数的原始实现
def sigmoid(input):
    y = [1/ float(1 + np.exp(-x)) for x in input]
    return y


def tanh(input):
    y = [np.exp(x) - np.exp(-x) -float(np.exp(x) - np.exp(-x)) for x in input]
    return y


def softpuls(input):
    y = [np.log(1 + np.exp(x)) for x in input]


#经过TensorFlow的激活函数处理的各个Y值

y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)


#创建会话
sess = tf.Session()

#运行
y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

plt.figure(1, figsize=(8, 6))

plt.subplot(221)
plt.plot(x, y_sigmoid, c = 'red', label = "sigmoid")
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_relu, c = 'red', label = "relu")
plt.ylim((-1, 6))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, c = 'red', label = "tanh")
plt.ylim((-1.3, 1.3))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, c = 'red', label = "softplus")
plt.ylim((-1, 6))
plt.legend(loc='best')

plt.show()

-----------------------------------CODE5------------------------------------------------





答辩准备：
一：以组为单位提交
		项目代码（训练数据，代码文件，演示文档，Markdown）
		PPT内容：功能介绍，实现
		Markdonw：组员分工
		压缩文件提交
二：流程
		1.分组演示介绍
		2.简单提问环节（3问）
		3.评分

