import tensorflow as tf
import numpy as np
import pandas as pd
import load_data as data

Xtr, Ytr = data.input_pipeline([data.train_file_name], data.train_batch_size,
								train_data=True, num_epochs=25)
Xte, Xte_id = data.input_pipeline([data.test_file_name], data.test_batch_size,
								train_data=False, num_epochs=25)

# Logistic Regression
xtr = tf.placeholder(tf.float32, [None, 6])
ytr = tf.placeholder(tf.float32, [None, 2])
w = tf.Variable(tf.zeros([6, 2]))
b = tf.Variable(tf.zeros([2]))

h = tf.nn.softmax(tf.matmul(xtr, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(ytr * tf.log(h), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_pred = tf.equal(tf.argmax(h, 1), tf.argmax(ytr, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	Xtr_list = None
	Ytr_list = None
	Xte_list = None
	Xte_id_list = None
	_y = []

	while True:
		try:
			Xtr_list, Ytr_list = sess.run([Xtr, Ytr])
			Xte_list, Xte_id_list = sess.run([Xte, Xte_id])
		except tf.errors.OutOfRangeError:
			break

		_y = []
		for i in range(len(Ytr_list)):
			val = Ytr_list[i]
			_y.append([val, 1 - val])

		sess.run(train_step, feed_dict={xtr: Xtr_list, ytr: _y})

	print sess.run(acc, feed_dict={xtr: Xtr_list, ytr: _y})