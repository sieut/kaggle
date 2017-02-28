import tensorflow as tf

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
    record_defaults = [[0], [-1.0], [-1.0], [-1.0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""], [0.0]]
    _, _, survived, p_class, _, sex, age, sibs_sp, par_ch, _, fare, _, _, f_size = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0.0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0.0)), lambda: tf.add(sex, tf.constant(1.0)))

    features = tf.stack([p_class, sex, age, sibs_sp, par_ch, fare, f_size])

    return features, survived

def read_test_data(name_queue):
    reader = tf.TextLineReader(skip_header_lines = 1)
    _, csv_row = reader.read(name_queue)
    record_defaults = [[0], [0], [-1.0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""], [0.0]]
    _, p_id, p_class, _, sex, age, sibs_sp, par_ch, _, fare, _, _, f_size = tf.decode_csv(csv_row, record_defaults=record_defaults)

    sex_comp = tf.equal(sex, "male")
    sex = 0.0
    tf.cond(sex_comp, lambda: tf.add(sex, tf.constant(0.0)), lambda: tf.add(sex, tf.constant(1.0)))

    features = tf.stack([p_class, sex, age, sibs_sp, par_ch, fare, f_size])

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

#Load train data
train_batch_size = file_len(train_file_name, skip_header_lines=1)
Xtr, Ytr = input_pipeline([train_file_name], train_batch_size, train_data=True, num_epochs=1)

#Load test data
test_batch_size = file_len(test_file_name, skip_header_lines=1)
Xte, Xte_id = input_pipeline([test_file_name], test_batch_size, train_data=False, num_epochs=1)