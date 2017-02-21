import tensorflow as tf
import numpy as np

train_file_name = "train_cleaned.csv"
test_file_name = "test_cleaned.csv"

def file_len(fname, skip_header_lines = 0):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1 - skip_header_lines

def read_train_data(name_queue):
    reader = tf.TextLineReader(skip_header_lines = 1)
    _, csv_row = reader.read(name_queue)
    record_defaults = [[0], [-1], [-1], [-1], [""], [""], [0.0], [0], [0], [""], [0.0], [""], [""]]
    _, _, survived, p_class, _, sex, age, sibs_sp, par_ch, _, _, _, _ = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0)), lambda: tf.add(sex, tf.constant(1)))

    features = tf.stack([p_class, sex, tf.to_int32(age), sibs_sp, par_ch])

    return features, survived

def read_test_data(name_queue):
    reader = tf.TextLineReader(skip_header_lines = 1)
    _, csv_row = reader.read(name_queue)
    record_defaults = [[0], [-1], [-1], [""], [""], [0.0], [0], [0], [""], [0.0], [""], [""]]
    _, _, p_class, _, sex, age, sibs_sp, par_ch, _, _, _, _ = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0)), lambda: tf.add(sex, tf.constant(1)))

    features = tf.stack([p_class, sex, tf.to_int32(age), sibs_sp, par_ch])

    return features

def input_pipeline(filenames, batch_size, train_data, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=num_epochs, shuffle=True)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size

    if train_data:
        features, survived = read_train_data(filename_queue)
        features_batch, label_batch = tf.train.batch(
        	[features, survived], batch_size=batch_size, capacity=capacity)
        return features_batch, label_batch
    else:
        features = read_test_data(filename_queue)
        features_batch = tf.train.batch(
            [features], batch_size=batch_size, capacity=capacity)
        return features_batch

#Loading train data
train_batch_size = file_len(train_file_name, skip_header_lines=1)
Xtr, Ytr = input_pipeline([train_file_name], train_batch_size, train_data=True, num_epochs=1)

#Loading test data
test_batch_size = file_len(test_file_name, skip_header_lines=1)
Xte = input_pipeline([test_file_name], test_batch_size, train_data=False, num_epochs=1)

#kNN
xte = tf.placeholder(tf.float32, [5])
xtr = tf.placeholder(tf.float32, [None, 5])
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1))
min_idx = tf.arg_min(distance, 0)

with tf.Session() as sess:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while True:
        Xtr_list = None
        Ytr_list = None
        Xte_list = None

        try:
            Xtr_list, Ytr_list = sess.run([Xtr, Ytr])
            Xte_list = sess.run([Xte])
        except tf.errors.OutOfRangeError:
            break

        for i in range(len(Xte_list[0])):
            knn_idx = sess.run(min_idx, feed_dict={xtr:Xtr_list, xte:Xte_list[0][0]})
            print Ytr_list[knn_idx]