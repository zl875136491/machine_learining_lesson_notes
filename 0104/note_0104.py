TensorFlow
	解释:
		|张量	|
		| ↓ 	|
		| ↓   	|   图
		| ↓     |
		|流动	|

		Server | Clients

	前端				| 		后端  
	编程模型			| 		运行 
	负责构造计算图	|		负责执行计算图
	多语言			|		C++

	TensorFlow库：
		1.TF 0.8
		2.AlphaGo 转向 TF
		3.语法解释器：SyntaxNet
		4.TF 0.9
		5.高层库：TF-Slim
		6.TF 1.0
		7.TF 1.3 : Estimator估算器
		8.TF 1.4 : Keras

	TF的特点：灵活，多语言，跨平台，高速

	TF的编程模式：
		一般编程模式分类：
			命令式编程			|		符号式编程
			python/JAVA/C++		|		图
			特点：基本无优化		|		特点：较多优化，嵌入
	
	张量：
	张量维度（秩）：Rank/Order
			|	维度		|	形式		|	类型		|
			|	0		|	1,2,3	|	标量		|
			|	1		|	[1,2.3]	|	向量		|
			|	2		|	[1,2,3]	|	矩阵		|
			|			|	[4,5,6] |			|
			|			|	[7,8,9] |			|
			|	3		|	......	|	3阶张量	|
			|	...		|	......	|	n阶张量	|

	张量属性：
	
		数据类型	dtype
			常量：tf.constant()
			变量：tf.Variable()
			占位：tf.placeholder()
			稀疏张量：tf.sparse()
		形状 Shape

		e·g：Tensor("Const:0", shape=(), dtype=int32)
	'''神经网络模拟网站：http://playground.tensorflow.org/'''

-----------------------------------CODE1------------------------------------------------

# -*- coding:UTF-8 -*-
import tensorflow as tf
#创建两个常量Tensor constant
const1 = tf.constant([[2,2]])
const2 = tf.constant([[4],[4]])
#张量相乘
multiply = tf.matmul(const1,const2)
print("sess.run()之前，尝试输出multiply的值：{}".format(multiply))

#第一种创建session对象
sess = tf.Session()
#用session的run方法来实际运行multiply这个矩阵乘法操作
#并且执行的结果赋值给result
result = sess.run(multiply)
#打印结果
print("sess.run()后，输出multiply：{}".format(result))
#关闭session对话
sess.close()


#第二种方法来创建和关闭Session
#用到了上下文管理器
with tf.Session() as sess:
	result2 = sess.run(multiply)
	print("结果是{}".format(result2))


-----------------------------------CODE1------------------------------------------------


-----------------------------------CODE2------------------------------------------------
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
    print("y = w * x + b, 值为：{}".format(result))
-----------------------------------CODE2------------------------------------------------


复制code2产生的log路径 xxx
在CMD中运行tensorboard服务：
tensorboard --logdir="xxx"
得到端口号X，在浏览器中访问localhost:X
（	如果tensorflow非全局包，则无法直接运行，
	在pycharm中找到该工程环境的Script包目录，
	在此目录中运行tensorboard		）