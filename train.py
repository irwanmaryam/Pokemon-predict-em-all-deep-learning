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


datadir = "/home/professor/Desktop/Irwan Project/image classification/PokemonData"
datadir = pathlib.Path(datadir)

image_count = len(list(datadir.glob('*/*.jpg')))
print(image_count)

classname = np.array([item.name for item in datadir.glob('*') if item.name!= "LICENSE.txt"])
print(classname)
label = list(classname)


batch_size = 32
img_height = 28
img_width = 28
epoch = 50

image_preprocess = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = image_preprocess.flow_from_directory(directory = str(datadir), batch_size = batch_size, target_size = (img_width, img_height), classes = label)


image, label = next(train_data)
X_train, X_test, y_train, y_test = train_test_split(image, label, test_size = 0.2, random_state = 42)

num_label = y_train.shape[1]

print(num_label)

model = cnn(num_label)

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


	zipped_data = zip(X_train, y_train)
	batches = helpers.batch_iter(list(zipped_data), batch_size, epoch)

	for batch in batches:


		Xbatch,Ybatch = zip(*batch)

		 


		train_dict = {model.x: Xbatch, model.y:Ybatch}

		_,step, loss, accuracy = sess.run([model.optimizer, model.global_step, model.cost, model.accuracy], feed_dict = train_dict)

		# train_write.add_summary(summary, step)

		if step % 1 == 0:


			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))

		# current_step = tf.train.global_step(sess, model.global_step)

		if step % 100 == 0:
			
			feed_dict = {model.x: xtest, model.y: ytest}

		
			step ,accuracy, loss= sess.run([model.global_step, model.accuracy, modelcost], feed_dict = feed_dict)
			# train_accuracy = sess.run(model.accuracy, feed_dict = feed_dict)
				
			print("Evaluate : ")
			# test_write.add_summary(summary, step)
			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))
			# print(matrix)

