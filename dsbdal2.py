import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/RohithYogi/Student-Performance-Prediction/master/input/students.csv")

df

df.isna().sum()

df.dtypes

df.shape

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(df['age'])

import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(df['age']))
df=df[(z<3)]
df.shape

sns.boxplot(df['age'])

x = df[['age']]

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(x)
x=norm.transform(x)
x