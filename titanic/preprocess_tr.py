import sys
import pandas as pd

df = pd.read_csv("train.csv")

# Drop N/A values in Age, Sex and Pclass
df = df.dropna(subset=['Age', 'Sex', 'Pclass'])

# Transform columns to values in range 0 - 1
df.Age = (df.Age - df.Age.min()) / (df.Age.max() - df.Age.min())
df.SibSp = (df.SibSp - df.SibSp.min()) / (df.SibSp.max() - df.SibSp.min())
df.Parch = (df.Parch - df.Parch.min()) / (df.Parch.max() - df.Parch.min())
df.Pclass = (df.Pclass - df.Pclass.min()) / (df.Pclass.max() - df.Pclass.min())
df.Fare = (df.Fare - df.Fare.min()) / (df.Fare.max() - df.Fare.min())

df.to_csv("train_cleaned.csv")