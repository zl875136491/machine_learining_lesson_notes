# -*- coding:UTF-8 -*-
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
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