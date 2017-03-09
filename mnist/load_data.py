import pandas as pd

train_file_name = "train.csv"
test_file_name = "test.csv"

def file_len(fname, skip_header_lines = 0):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1 - skip_header_lines

def read_tr():
	df = pd.read_csv(train_file_name)
	labels = df['label']
	pixels = []

	for i in range(len(df)):
		pixels.append(df.loc[i].drop('label'))

	return pixels, labels