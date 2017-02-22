import tensorflow as tf
import numpy as np
import pandas as pd

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
    record_defaults = [[0], [-1.0], [-1.0], [-1.0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]
    _, _, survived, p_class, _, sex, age, sibs_sp, par_ch, _, _, _, _ = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0.0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0.0)), lambda: tf.add(sex, tf.constant(1.0)))

    features = tf.stack([p_class, sex, age, sibs_sp, par_ch])

    return features, survived

def read_test_data(name_queue):
    reader = tf.TextLineReader(skip_header_lines = 1)
    _, csv_row = reader.read(name_queue)
    record_defaults = [[0.0], [-1.0], [-1.0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]]
    p_id, _, p_class, _, sex, age, sibs_sp, par_ch, _, _, _, _ = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0.0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0.0)), lambda: tf.add(sex, tf.constant(1.0)))

    features = tf.stack([p_class, sex, age, sibs_sp, par_ch])

    return features, p_id

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
        features, p_id = read_test_data(filename_queue)
        features_batch, id_batch = tf.train.batch(
            [features, p_id], batch_size=batch_size, capacity=capacity)
        return features_batch, id_batch

#Loading train data
train_batch_size = file_len(train_file_name, skip_header_lines=1)
Xtr, Ytr = input_pipeline([train_file_name], train_batch_size, train_data=True, num_epochs=1)

#Loading test data
test_batch_size = file_len(test_file_name, skip_header_lines=1)
Xte, Xte_id = input_pipeline([test_file_name], test_batch_size, train_data=False, num_epochs=1)

#kNN
xte = tf.placeholder(tf.float32, [5])
xtr = tf.placeholder(tf.float32, [None, 5])
k = tf.placeholder(tf.int32, shape=())
distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1))
inverse_distance = tf.divide(tf.constant(1.0), distance)
_, min_dist_idx = tf.nn.top_k(inverse_distance, k)

knn_survived = tf.placeholder(tf.float32, [None])
pred = tf.to_int32(tf.round(tf.reduce_mean(knn_survived)))

def train_best_k():
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

            accuracy = []
            for k_te in range(1,11):
                current_accuracy = 0
                for i in range(len(Xtr_list)):
                    x_te = Xtr_list[i]
                    knn_idx = sess.run(min_dist_idx, feed_dict={xtr:Xtr_list, xte:x_te, k: k_te + 1})
                    if knn_idx[0] != i:
                        knn_idx = knn_idx[:-1]
                    else:
                        knn_idx = knn_idx[1:]
                    surv = sess.run(pred, feed_dict={knn_survived: Ytr_list[knn_idx]})
                    if int(surv) - Ytr_list[i] == 0:
                        current_accuracy = current_accuracy + 1

                accuracy.append(float(current_accuracy) / float(len(Xtr_list)))

            print accuracy

def test_data():
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
                Xtr_list, Ytr_list = sess.run([Xtr, Ytr])
                Xte_list, Xte_id_list = sess.run([Xte, Xte_id])
            except tf.errors.OutOfRangeError:
                break

            pred_list = []
            for i in range(len(Xte_list)):
                knn_idx = sess.run(min_dist_idx, feed_dict={xtr:Xtr_list, xte:Xte_list[i], k: 5})
                pred_list.append(sess.run(pred, feed_dict={knn_survived: Ytr_list[knn_idx]}))

            df = pd.DataFrame({"PassengerId": Xte_id_list, "Survived": pred_list})
            df.to_csv("prediction.csv")

test_data()