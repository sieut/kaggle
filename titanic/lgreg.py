import tensorflow as tf
import numpy as np
import pandas as pd
import load_data as data

# Logistic Regression
xtr = tf.placeholder(tf.float32, [None, 5])
ytr = tf.placeholder(tf.float32, [None, 1])
w = tf.placeholder(tf.float32, [5])

z = tf.reduce_sum(tf.multiply(xtr, w), reduction_indices=1)
g = tf.divide(tf.constant(1.0), tf.constant(1.0) + tf.exp(tf.negative(z)))

with tf.Session() as sess:
	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	while True:
		Xtr_list = None
		Ytr_list = None
		Xte_list = None
		Xte_id_list = None

		try:
			Xtr_list, Ytr_list = sess.run([data.Xtr, data.Ytr])
			Xte_list, Xte_id_list = sess.run([data.Xte, data.Xte_id])
		except tf.errors.OutOfRangeError:
			break

		w_tr = [1, 1, 1, 1, 1]
		print sess.run(g, feed_dict={xtr: Xtr_list, w: w_tr})