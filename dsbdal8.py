import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df=pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df

df.shape

sns.distplot(df['Pclass'])

sns.distplot(df['Age'])

sns.distplot(df['Age'], bins=40)

sns.jointplot(x=df['Age'], y=df['Fare'], kind="scatter")

sns.pairplot(df)

sns.barplot(x=df['Sex'],y=df['Fare'])

sns.countplot(df['Pclass'])

sns.histplot(data=df, x="Fare",y="PassengerId"  ,kde=False)