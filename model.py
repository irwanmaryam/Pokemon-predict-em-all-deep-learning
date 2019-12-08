import tensorflow as tf
import os
import time
import sys
import numpy as np
from tensorflow.python.platform import gfile

class cnn(object):
	def __init__(self, img_size, num_label):


		

		self.global_step = tf.Variable(0, name = "global_step")



		self.x = tf.placeholder(tf.float32, [None, img_size, img_size, 3], name = 'x') 

		# print(self.x)


		# self.x_image = tf.reshape(self.x, [-1, img_size, img_size, 1], name= 'input')

		# print(self.x_image)

		self.y =  tf.placeholder(tf.float32, shape=(None, num_label), name='output_y')

		# self.y_cls = tf.argmax(y_true, dimension = 1)

		# print(self.y_cls)

		weight1 = tf.Variable(tf.truncated_normal([2,2,3, 32], stddev = 0.1), name = 'weightcnn')

		bias1 = tf.Variable(tf.zeros([32]))

		self.convolutional1 = tf.nn.conv2d(self.x, weight1, [1, 1, 1, 1], padding = 'VALID')

		print(self.convolutional1)

		self.relu = tf.nn.relu(self.convolutional1) + bias1

		self.pooling1 = tf.nn.max_pool(self.relu, [1,2,2,1], [1, 2, 2, 1], padding = 'SAME')

		weight2 = tf.Variable(tf.truncated_normal([3,3,32, 64], stddev = 0.1), name = 'weightcnn2')

		bias2 = tf.Variable(tf.zeros([64]))

		self.convolutional2 = tf.nn.conv2d(self.pooling1, weight2, [1,2,2,1], padding = 'VALID')

		print(self.convolutional2)

		self.relu2 = tf.nn.relu(self.convolutional2) + bias2

		self.pooling2 = tf.nn.max_pool(self.relu2, [1,4,4,1], [1,4,4,1], padding = 'SAME', name = 'max_pool2')

		print(self.pooling2)


		# flat = tf.contrib.layers.flatten(self.pooling2)


		# flat = tf.reshape(self.pooling2, [-1, 3*3*64])

		flat = self.pooling2.get_shape().as_list()



		# num_feature = flat[0] * flat[1] * flat[2]

		num_feature = tf.reshape(self.pooling2, [-1, int(flat[1]) * int(flat[2]) * int(flat[3])])

		print(num_feature)

		# weightfc = tf.Variable(tf.truncated_normal([num_feature, num_label], stddev = 0.5), name = 'weightfc')

		# biasfc = tf.Variable(tf.zeros([num_label]), name='biasfc')

		# self.fc1 = tf.matmul(num_feature, weightfc) + biasfc



		bath1 = tf.layers.batch_normalization(num_feature)
		self.fc1 = tf.layers.dense(bath1, 64)
		bath2 = tf.layers.batch_normalization(self.fc1)
		self.fc2 = tf.layers.dense(bath2, num_label)

		# self.fc2 = tf.layers.dense(self.fc1, num_label)

		
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.y))

		print("loss : ", self.cost)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.cost)

		# Accuracy
		self.correct_pred = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))

		print("predicted labels : ", self.correct_pred)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')


