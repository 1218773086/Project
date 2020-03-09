import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


filename = os.listdir("./train datasets/image")
new_dir  = "./train datasets/"
x_input =[]
y_input =[]

for img in filename:
    img=os.path.splitext(img)[0]
    xs = pd.read_csv(new_dir + img + '.csv',index_col=0)
    ys = pd.read_csv(new_dir + img + '.dat',usecols = [1],sep='\s+')
    xs = xs.values.flatten()
    ys = ys.values.flatten()
    xs = np.expand_dims(xs,0)
    ys = np.expand_dims(ys,0)
    x_input.append(xs)
    y_input.append(ys)
x_train, x_test, y_train, y_test = train_test_split(x_input,y_input,test_size=100, random_state=0)




x = tf.placeholder("float", shape=[None, 30000], name='x')  # 输入
y_ = tf.placeholder("float", shape=[None, 49], name='y_')  # 实际值


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 产生正态分布 标准差0.1
    return tf.Variable(initial)
# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 定义常量
    return tf.Variable(initial)

# 卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')  # 最大池化

x_image = tf.reshape(x, [-1, 200, 150, 1])

# 第一层卷积
W_conv1 = weight_variable([4, 4, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积
W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# 第四层卷积
W_conv4 = weight_variable([2, 2, 128, 128])
b_conv4 = bias_variable([128])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

# 全连接层1
W_fc1 = weight_variable([16640, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool4, [-1, 16640])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# dropout 防止过拟合
keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#全连接层2
W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# dropout 防止过拟合
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# 输出层
W_fc3 = weight_variable([256, 49])
b_fc3 = bias_variable([49])

y_conv = tf.nn.xw_plus_b(h_fc2_drop, W_fc3 , b_fc3,name='y_conv')

# 训练和评估模型
loss = tf.reduce_mean(tf.square(y_ - y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./')
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess,ckpt.model_checkpoint_path)

    for i in range(10):
        sum_test=0
        sum_train=0
        for j in range(0,len(x_train)-1):
            train_step.run(feed_dict={x: x_train[j], y_: y_train[j], keep_prob: 0.5})
            mse_train=sess.run(loss, feed_dict={x: x_train[j], y_: y_train[j], keep_prob: 1.0})
            #print(sess.run(y_conv,feed_dict={x:x_train[j],keep_prob:1.0}))
            sum_train+=mse_train
        if i % 1 == 0:
            print(sum_train/len(x_train),end='  ')
        for m in range(100):
            mse_test=sess.run(loss,feed_dict={x : x_test[m], y_: y_test[m], keep_prob: 1.0})
            sum_test+=mse_test
        print(sum_test/100)
        # 保存模型

    saver.save(sess, './model.ckpt')





