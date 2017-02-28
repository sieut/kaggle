import tensorflow as tf
import pandas as pd
import load_data as data

#kNN
xte = tf.placeholder(tf.float32, [7])
xtr = tf.placeholder(tf.float32, [None, 7])
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
                Xtr_list, Ytr_list = sess.run([data.Xtr, data.Ytr])
                Xte_list = sess.run([data.Xte])
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
                Xtr_list, Ytr_list = sess.run([data.Xtr, data.Ytr])
                Xte_list, Xte_id_list = sess.run([data.Xte, data.Xte_id])
            except tf.errors.OutOfRangeError:
                break

            pred_list = []
            for i in range(len(Xte_list)):
                knn_idx = sess.run(min_dist_idx, feed_dict={xtr:Xtr_list, xte:Xte_list[i], k: 5})
                pred_list.append(sess.run(pred, feed_dict={knn_survived: Ytr_list[knn_idx]}))

            df = pd.DataFrame({"PassengerId": Xte_id_list, "Survived": pred_list})
            df.set_index("PassengerId", inplace=True)
            df.to_csv("prediction.csv")

test_data()