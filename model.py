import tensorflow as tf
import os
import time
import sys
import numpy as np
from tensorflow.python.platform import gfile

class cnn(object):
	def __init__(self, num_label):


		kernel_size1 = 1
		num_kernel1 = 64
		pooling = 1

		kernel_size2 = 2
		num_kernel2 = 32
		pooling = 2

		# channel/ = 3


		img_size = 28

		img_size_flat = img_size * img_size 

		img_shape = (img_size, img_size)

		batch_size = 32

		# validation split = 0.1


		self.global_step = tf.Variable(0, name = "global_step")



		self.x = tf.placeholder(tf.float32, [None, 28,28, 1], name = 'x') 

		print(self.x)


		self.x_image = tf.reshape(self.x, [-1, img_size, img_size, 1], name= 'input')

		print(self.x_image)

		self.y =  tf.placeholder(tf.float32, shape=(None, num_label), name='output_y')



		weight1 = tf.Variable(tf.truncated_normal([1,1,1, 32]))

		bias1 = tf.Variable(tf.zeros([32]))

		self.convolutional1 = tf.nn.conv2d(self.x_image, weight1, [1, 1, 1, 1], padding = 'VALID')

		print(self.convolutional1)

		self.relu = tf.nn.relu(self.convolutional1) + bias1

		self.pooling1 = tf.nn.max_pool(self.relu, [1,2,2,1], [1, 2, 2, 1], padding = 'SAME')

		weight2 = tf.Variable(tf.truncated_normal([3,3,32, 64]))

		bias2 = tf.Variable(tf.zeros([64]))

		self.convolutional2 = tf.nn.conv2d(self.pooling1, weight2, [1,2,2,1], padding = 'VALID')

		print(self.convolutional2)

		self.relu2 = tf.nn.relu(self.convolutional2) + bias2

		self.pooling2 = tf.nn.max_pool(self.relu2, [1,3,3,1], [1,3,3,1], padding = 'SAME')

		print(self.pooling2)


		# flat = tf.contrib.layers.flatten(self.pooling2)


		flat = tf.reshape(self.pooling2, [-1, 28*28*64])


		self.fc1 = tf.layers.dense(flat, 64)

		self.fc2 = tf.layers.dense(self.fc1, num_label)

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.y))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.cost)

		# Accuracy
		self.correct_pred = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')


