import tensorflow as tf

train_file_name = "train.csv"
test_file_name = "test.csv"

def file_len(fname, skip_header_lines = 0):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1 - skip_header_lines

def features_check(features):
    null_features = tf.constant([-1, -1, 0, -1, -1])
    compare_null_features = tf.add(features, tf.negative(null_features))

    no_diff = tf.zeros([1, 5], tf.int32)
    _, idx = tf.setdiff1d(compare_null_features, no_diff)

    cond = tf.not_equal(idx.get_shape(), tf.TensorShape([0,0]))
    return cond


def read_from_csv(name_queue):
    reader = tf.TextLineReader(skip_header_lines = 1)
    _, csv_row = reader.read(name_queue)
    record_defaults = [[-1], [-1], [-1], [""], [""], [0.0], [0], [0], [""], [0.0], [""], [""]]
    _, survived, p_class, _, sex, age, sibs_sp, par_ch, _, _, _, _ = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0)), lambda: tf.add(sex, tf.constant(1)))

    features = tf.pack([p_class, sex, tf.to_int32(age), sibs_sp, par_ch])

    #check = features_check(features)

    return features, survived

def input_pipeline(filenames, batch_size, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=num_epochs, shuffle=True)
	features, survived = read_from_csv(filename_queue)

	min_after_dequeue = 10000
	capacity = min_after_dequeue + 3 * batch_size

	example_batch, label_batch = tf.train.batch(
		[features, survived], batch_size=batch_size, capacity=capacity)
		#min_after_dequeue=min_after_dequeue)
	return example_batch, label_batch

train_batch_size = file_len(train_file_name, skip_header_lines=1)
X, y = input_pipeline([train_file_name], train_batch_size, 1)

with tf.Session() as sess:
    tf.initialize_local_variables().run()
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while True:
        try:
            Xtr, ytr = sess.run([X, y])
            print (Xtr, ytr)
        except tf.errors.OutOfRangeError:
            break