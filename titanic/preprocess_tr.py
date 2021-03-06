import sys
import pandas as pd

df = pd.read_csv("train.csv")

def set_average_age_for_title(title):
	idx = df.Name.str.contains(title)
	df_contain_title = df.Age[idx]
	df_contain_title = df_contain_title.dropna()
	df.loc[(idx) & (df.Age.isnull()), "Age"] = df_contain_title.mean()

titles = ["Mr\.", "Mrs\.", "Miss\.", "Master\."]

for title in titles:
	set_average_age_for_title(title)

# Drop N/A values in Age, Sex and Pclass
df = df.dropna(subset=['Age', 'Sex', 'Pclass'])

# Add new variable, Family Size
df['Fsize'] = df.SibSp + df.Parch + 1

# Transform columns to values in range 0 - 1
df.Age = (df.Age - df.Age.min()) / (df.Age.max() - df.Age.min())
df.SibSp = (df.SibSp - df.SibSp.min()) / (df.SibSp.max() - df.SibSp.min())
df.Parch = (df.Parch - df.Parch.min()) / (df.Parch.max() - df.Parch.min())
df.Pclass = (df.Pclass - df.Pclass.min()) / (df.Pclass.max() - df.Pclass.min())
df.Fare = (df.Fare - df.Fare.min()) / (df.Fare.max() - df.Fare.min())
df.Fsize = (df.Fsize - df.Fsize.min()) / (df.Fsize.max() - df.Fsize.min())

df.to_csv("train_cleaned.csv")