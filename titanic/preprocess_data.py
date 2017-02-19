import sys
import pandas as pd

if len(sys.argv) > 1:
	df = pd.read_csv(sys.argv[1])
	df = df.dropna(subset=['Age', 'Sex', 'Pclass'])
	df.to_csv(sys.argv[1][:-4] + '_cleaned.csv')