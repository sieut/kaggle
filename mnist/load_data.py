import pandas as pd

train_file_name = "train.csv"
test_file_name = "test.csv"

def read_tr():
	df = pd.read_csv(train_file_name)

	labels = df['label'].tolist()
	labels_ytr = []
	for label in labels:
		label_ytr = [0] * 10
		label_ytr[label] = 1
		labels_ytr.append(label_ytr)

	pixels = []
	for i in range(len(df)):
		pixels.append(df.loc[i].drop('label').tolist())

	return pixels, labels_ytr