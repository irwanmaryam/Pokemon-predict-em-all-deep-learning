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


iteration = 10
batch_size = 25
img_size = 28
channel = 3


datadir = "/home/professor/Desktop/Irwan Project/image classification/PokemonData"
validation = 0.2


category = os.listdir(datadir)
num_label = len(category)

data = helpers.read_train_sets(datadir,img_size, category, validation_size = validation )

print("train set:		".format(len(data.train.labels)))
print("test set 		".format(len(data.valid.labels)))


model = cnn(img_size, num_label)

#### model CNN #####



print("training started")
with tf.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	saver = tf.compat.v1.train.Saver(tf.global_variables())


	# tf.summary.text("text", b)

	# tf.summary.histogram("weights", weight)
	# tf.summary.histogram("fc1", model.fc)
	# tf.summary.histogram("fc2", model.fc3)
	# tf.summary.histogram("fc3", model.fc3)

	# tf.summary.scalar("accuracy", accuracy)
	# tf.summary.scalar("loss", loss)
	# merge = tf.summary.merge_all()

	# train_write = tf.summary.FileWriter(logdir + "/train", sess.graph)
	# test_write = tf.summary.FileWriter(logdir + "/test", sess.graph)

	# tf.global_variables_initializer().run()

	step = 0
	for i in range(iteration):

		Xbatch, Ybatch, _, cls_batch = data.train.next_batch(batch_size)
		xValid, yValid, _, validcls_batch = data.valid.next_batch(batch_size)


		train_dict = {model.x: Xbatch, model.y: Ybatch}

		test_dict = {model.x: xValid, model.y:yValid}

		_, step, loss, accuracy = sess.run([model.optimizer, model.global_step, model.cost, model.accuracy], feed_dict = train_dict)


		if step % 1 == 0:


			
			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))

		if step % 100 == 0:

				
			# feed_dict = {model.x: xtest, model.y: ytest, model.keep_drop:1.0}


			step ,accuracy, loss= sess.run([model.global_step, model.accuracy, model.cost], feed_dict = test_dict)
			# train_accuracy = sess.run(model.accuracy, feed_dict = feed_dict)
				
			print("Evaluate : ")
			# test_write.add_summary(summary, step)
			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))
			# print(matrix)
