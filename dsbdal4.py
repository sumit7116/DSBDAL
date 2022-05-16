import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
df

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df.info()

df.describe().transpose()

df.isna().sum()

df.columns

df.shape

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot =True)

x=df[['rm','lstat']]
y=df['medv']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

m1 = LinearRegression()

m1.fit(x_train,y_train)
y_pred = m1.predict(x_test)

print("Model Score: ", m1.score(x_test,y_test))