####Assignment 8 ######
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


####Assignment 9 ######
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df

df.shape

df.isna().sum()

df['Age'] = df['Age'].fillna(df['Age'].mean())

df.isna().sum()

import matplotlib.pyplot as plt
import seaborn as sns

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = df[df['Sex']=='female']
men = df[df['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

sns.barplot(x='Pclass', y='Survived', data=df)

from plotnine.data import economics
from plotnine import ggplot, aes, geom_bar

ggplot(aes(x="Sex", fill="Sex"), df) + geom_bar()

ggplot(aes(x = "Survived"),df)+geom_bar()

ggplot(aes("Sex", fill = "Survived"),df)+geom_bar()

ggplot(aes("Sex", fill = "Survived"),df)+geom_bar()


####Assignment 10 ######
import numpy as np
import pandas as pd

df = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
df

column = len(list(df))
column

df.info()

df.head()

np.unique(df["species"])

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

fig, axes = plt.subplots(2, 2, figsize=(16, 8))


axes[0,0].set_title("Distribution of First Column")
axes[0,0].hist(df["sepal_length"]);

axes[0,1].set_title("Distribution of Second Column")
axes[0,1].hist(df["sepal_width"]);

axes[1,0].set_title("Distribution of Third Column")
axes[1,0].hist(df["petal_length"]);

axes[1,1].set_title("Distribution of Fourth Column")
axes[1,1].hist(df["petal_width"]);

df.dtypes

df['sepal_length'].value_counts()

data_to_plot = [df["sepal_length"],df["sepal_width"],df["petal_length"],df["petal_width"]]

sns.set_theme(style="whitegrid")
# Creating a figure instance
fig = plt.figure(1, figsize=(12,8))

# Creating an axes instance
ax = fig.add_subplot(111)

# Creating the boxplot
bp = ax.boxplot(data_to_plot)
