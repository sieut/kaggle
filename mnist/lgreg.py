import tensorflow as tf
import load_data as data

Xtr = tf.placeholder(tf.float32, [None, 784])
Ytr = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(Xtr, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(Ytr * tf.log(pred), reduction_indices=1))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
	tf.local_variables_initializer().run()
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	xtr, ytr = data.read_tr()

	for _ in range(25):
		sess.run(train_step, feed_dict={Xtr: xtr, Ytr: ytr})

	print w.eval()
	print b.eval()