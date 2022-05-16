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