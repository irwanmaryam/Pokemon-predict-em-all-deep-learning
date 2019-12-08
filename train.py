from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import IPython.display as display
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pathlib
import glob
import helpers
from model import cnn


iteration = 1000
batch_size = 32
img_size = 128
channel = 3


datadir = "/home/professor/Desktop/Irwan Project/image classification/PokemonData"
validation = 0.2


category = os.listdir(datadir)
num_label = len(category)

data = helpers.read_train_sets(datadir,img_size, category, validation_size = validation )

print("train set:{}		".format(len(data.train.labels)))
print("test set :{}		".format(len(data.valid.labels)))


model = cnn(img_size, num_label)


logdir = "./logs/nn_logs"



#### model CNN #####



print("training started")
with tf.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	saver = tf.compat.v1.train.Saver(tf.global_variables())


	tf.summary.scalar("accuracy", model.accuracy)
	tf.summary.scalar("loss", model.cost)
	merge = tf.summary.merge_all()

	train_write = tf.summary.FileWriter(logdir + "/train", sess.graph)
	test_write = tf.summary.FileWriter(logdir + "/test", sess.graph)

	tf.global_variables_initializer().run()


	
	for i in range(iteration):

		
		Xbatch, Ybatch, _, cls_batch = data.train.next_batch(batch_size)
		xValid, yValid, _, validcls_batch = data.valid.next_batch(batch_size)

		
		# print("step : ", i + 1)

		train_dict = {model.x : Xbatch, model.y:Ybatch}

		_,summary, accuracy, loss = sess.run([model.optimizer, merge, model.accuracy, model.cost], feed_dict = train_dict)

		print("step : {0} loss = {1:.3f} accuracy = {2:.3f}".format(i+1, loss, accuracy))

		if i % 100 == 0:
			test_dict = {model.x:xValid, model.y:yValid}

			summarry, accuracy, loss = sess.run([merge, model.accuracy, model.cost], feed_dict = test_dict)

			print("evaluation")

			print("step : {0} loss = {1:.3f} accuracy = {2:.3f}".format(i, loss, accuracy))
	print("finish")