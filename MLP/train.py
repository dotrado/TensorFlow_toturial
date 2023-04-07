# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:52:57 2018
@author: GY
"""
import inference
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import os


BATCH_SIZE=100
TRANING_STEPS=10000
INPUT_NODE=784
OUTPUT_NODE=10
IMAGE_SIZE1=28
IMAGE_SIZE2=28
NUM_CHANNELS=1


model_dir = "saver"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
MODEL_SAVE_PATH=model_dir
MODEL_NAME="model.ckpt"
def train(mnist):

    x=tf.placeholder (tf.float32,[None,INPUT_NODE],name='x-input')

    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    y=inference.inference(x,keep_prob=0.5)
    global_step=tf.Variable(0,trainable=False)
    cross_entropy=-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])

    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean
    train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy_mean, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_step]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRANING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100==0:
                train_accuracy = accuracy.eval(feed_dict={x: xs, y_: ys})
                print ("After %d training steps, loss %g, training accuracy %g"%(step,loss_value,train_accuracy))
                # print("predict:", sess.run([tf.argmax(y, 1)], feed_dict={x: reshaped_xs}))
                # print("labels:", np.argmax(ys, 1))
                # print("cross_entropy_mean", sess.run([cross_entropy_mean], feed_dict={x: reshaped_xs, y_: ys}))
                # print("losses", sess.run(tf.add_n(tf.get_collection('losses')), feed_dict={x: reshaped_xs, y_: ys}))
                # print ("learning rate",sess.run(learning_rate))
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(argv=None):
    mnist=read_data_sets('MNIST_data', one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()


##TODO: Training process
# After 1 training steps, loss 3.1308, training accuracy 0.09
# After 101 training steps, loss 0.734694, training accuracy 0.75
# After 201 training steps, loss 0.691555, training accuracy 0.72
# After 301 training steps, loss 0.386326, training accuracy 0.93
# After 401 training steps, loss 0.571477, training accuracy 0.87
# After 501 training steps, loss 0.405222, training accuracy 0.9
# After 601 training steps, loss 0.322248, training accuracy 0.88
# After 701 training steps, loss 0.270225, training accuracy 0.9
# After 801 training steps, loss 0.298285, training accuracy 0.91
# After 901 training steps, loss 0.269336, training accuracy 0.96
# After 1001 training steps, loss 0.294913, training accuracy 0.87
# After 1101 training steps, loss 0.196081, training accuracy 0.96
# After 1201 training steps, loss 0.251217, training accuracy 0.89
# After 1301 training steps, loss 0.171882, training accuracy 0.92
# After 1401 training steps, loss 0.199019, training accuracy 0.97
# After 1501 training steps, loss 0.182819, training accuracy 0.95
# After 1601 training steps, loss 0.260835, training accuracy 0.95
# After 1701 training steps, loss 0.129049, training accuracy 0.95
# After 1801 training steps, loss 0.315554, training accuracy 0.91
# After 1901 training steps, loss 0.155339, training accuracy 0.93
# After 2001 training steps, loss 0.260566, training accuracy 0.91
# After 2101 training steps, loss 0.184313, training accuracy 0.95
# After 2201 training steps, loss 0.183568, training accuracy 0.96
# After 2301 training steps, loss 0.228263, training accuracy 0.91
# After 2401 training steps, loss 0.252987, training accuracy 0.92
# After 2501 training steps, loss 0.148908, training accuracy 0.95
# After 2601 training steps, loss 0.112555, training accuracy 0.98
# After 2701 training steps, loss 0.196327, training accuracy 0.97
# After 2801 training steps, loss 0.143428, training accuracy 0.97
# After 2901 training steps, loss 0.153906, training accuracy 0.95
# After 3001 training steps, loss 0.115412, training accuracy 0.97
# After 3101 training steps, loss 0.200469, training accuracy 0.94
# After 3201 training steps, loss 0.193037, training accuracy 0.97
# After 3301 training steps, loss 0.196158, training accuracy 0.97
# After 3401 training steps, loss 0.164673, training accuracy 0.96
# After 3501 training steps, loss 0.269091, training accuracy 0.93
# After 3601 training steps, loss 0.181332, training accuracy 0.95
# After 3701 training steps, loss 0.157314, training accuracy 0.97
# After 3801 training steps, loss 0.0872078, training accuracy 0.95
# After 3901 training steps, loss 0.107228, training accuracy 0.96
# After 4001 training steps, loss 0.115001, training accuracy 0.94
# After 4101 training steps, loss 0.0795502, training accuracy 0.95
# After 4201 training steps, loss 0.211854, training accuracy 0.95
# After 4301 training steps, loss 0.138561, training accuracy 0.97
# After 4401 training steps, loss 0.0969866, training accuracy 0.98
# After 4501 training steps, loss 0.14253, training accuracy 0.95
# After 4601 training steps, loss 0.190279, training accuracy 0.94
# After 4701 training steps, loss 0.180761, training accuracy 0.95
# After 4801 training steps, loss 0.115053, training accuracy 0.96
# After 4901 training steps, loss 0.155104, training accuracy 0.94
# After 5001 training steps, loss 0.12347, training accuracy 0.97
# After 5101 training steps, loss 0.13618, training accuracy 0.96
# After 5201 training steps, loss 0.0587247, training accuracy 0.99
# After 5301 training steps, loss 0.108271, training accuracy 1
# After 5401 training steps, loss 0.0878254, training accuracy 0.99
# After 5501 training steps, loss 0.107574, training accuracy 0.97
# After 5601 training steps, loss 0.0654617, training accuracy 0.99
# After 5701 training steps, loss 0.0778542, training accuracy 0.97
# After 5801 training steps, loss 0.0838198, training accuracy 0.94
# After 5901 training steps, loss 0.0847692, training accuracy 0.99
# After 6001 training steps, loss 0.0668485, training accuracy 0.97
# After 6101 training steps, loss 0.153695, training accuracy 0.97
# After 6201 training steps, loss 0.0586072, training accuracy 0.99
# After 6301 training steps, loss 0.105008, training accuracy 0.99
# After 6401 training steps, loss 0.0473971, training accuracy 0.98
# After 6501 training steps, loss 0.0513671, training accuracy 0.97
# After 6601 training steps, loss 0.0731774, training accuracy 0.96
# After 6701 training steps, loss 0.0445081, training accuracy 0.96
# After 6801 training steps, loss 0.0560657, training accuracy 0.99
# After 6901 training steps, loss 0.0206276, training accuracy 0.99
# After 7001 training steps, loss 0.0643102, training accuracy 0.97
# After 7101 training steps, loss 0.103166, training accuracy 0.96
# After 7201 training steps, loss 0.0521504, training accuracy 0.96
# After 7301 training steps, loss 0.0565876, training accuracy 0.96
# After 7401 training steps, loss 0.105062, training accuracy 0.99
# After 7501 training steps, loss 0.123677, training accuracy 0.96
# After 7601 training steps, loss 0.0609665, training accuracy 0.99
# After 7701 training steps, loss 0.0435051, training accuracy 0.98
# After 7801 training steps, loss 0.114205, training accuracy 0.96
# After 7901 training steps, loss 0.142729, training accuracy 0.98
# After 8001 training steps, loss 0.0859795, training accuracy 0.95
# After 8101 training steps, loss 0.0392768, training accuracy 1
# After 8201 training steps, loss 0.204215, training accuracy 0.96
# After 8301 training steps, loss 0.0518152, training accuracy 0.98
# After 8401 training steps, loss 0.048407, training accuracy 1
# After 8501 training steps, loss 0.112105, training accuracy 0.93
# After 8601 training steps, loss 0.0700119, training accuracy 0.95
# After 8701 training steps, loss 0.0350824, training accuracy 0.99
# After 8801 training steps, loss 0.0648541, training accuracy 0.97
# After 8901 training steps, loss 0.0354921, training accuracy 0.98
# After 9001 training steps, loss 0.0377504, training accuracy 0.99
# After 9101 training steps, loss 0.0582033, training accuracy 0.97
# After 9201 training steps, loss 0.0790014, training accuracy 0.98
# After 9301 training steps, loss 0.0906473, training accuracy 0.97
# After 9401 training steps, loss 0.120886, training accuracy 0.97
# After 9501 training steps, loss 0.0408178, training accuracy 0.99
# After 9601 training steps, loss 0.0215761, training accuracy 1
# After 9701 training steps, loss 0.0712243, training accuracy 0.98
# After 9801 training steps, loss 0.137363, training accuracy 0.98
# After 9901 training steps, loss 0.0941356, training accuracy 0.97

# -*-coding: utf-8 -*-
"""
    @Project: tensorflow_models_nets
    @File   : convert_pb.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-08-29 17:46:50
    @info   :
    -通过传入 CKPT 模型的路径得到模型的图和变量数据
    -通过 import_meta_graph 导入模型中的图
    -通过 saver.restore 从模型中恢复图中各个变量的数据
    -通过 graph_util.convert_variables_to_constants 将模型持久化
"""

import tensorflow as tf
from create_tf_record import *
from tensorflow.python.framework import graph_util

resize_height = 299  # 指定图片高度
resize_width = 299  # 指定图片宽度
depths = 3


def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")

            # 读取测试图片
            im = read_image(image_path, resize_height, resize_width, normalization=True)
            im = im[np.newaxis, :]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out = sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                          input_keep_prob_tensor: 1.0,
                                                          input_is_training_tensor: False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print
            "pre class_id:{}".format(sess.run(class_id))


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in sess.graph.get_operations():
        #     print(op.name, op.values())


def freeze_graph2(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

        # for op in graph.get_operations():
        #     print(op.name, op.values())


if __name__ == '__main__':
    # 输入ckpt模型路径
    input_checkpoint = 'models/model.ckpt-10000'
    # 输出pb模型的路径
    out_pb_path = "models/pb/frozen_model.pb"
    # 调用freeze_graph将ckpt转为pb
    freeze_graph(input_checkpoint, out_pb_path)

    # 测试pb模型
    image_path = 'test_image/animal.jpg'
    freeze_graph_test(pb_path=out_pb_path, image_path=image_path)
————————————————
版权声明：本文为CSDN博主「AI吃大瓜」的原创文章，遵循CC
4.0
BY - SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https: // blog.csdn.net / guyuealian / article / details / 82218092