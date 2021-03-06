import pandas as pd
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
df.sample(10)

df.isna().sum()

df.isnull().values.any()

df[(df['sepal_length'] < 0 )]

import numpy as np
import scipy.stats as stats
z=np.abs(stats.zscore(df['sepal_length']))
df=df[(z<3)]
df.shape

z=np.abs(stats.zscore(df['sepal_width']))
df=df[(z<3)]
df.shape

z=np.abs(stats.zscore(df['petal_length']))
df=df[(z<3)]
df.shape

z=np.abs(stats.zscore(df['petal_width']))
df=df[(z<3)]
df.shape

X = df[['petal_length','sepal_length','petal_width']]
Y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, y_pred)
print(cm)

import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)

def myscores(smat): 
    tp = smat[0][0] 
    fp = smat[0][1] 
    fn = smat[1][0] 
    tn = smat[1][1] 
    return tp/(tp+fp), tp/(tp+fn)

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(Y_test, y_pred)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

from sklearn.metrics import f1_score
f1_score(Y_test, y_pred, average=None)

import seaborn as sns
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"sepal_length").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"sepal_width").add_legend()
plt.show()

X = df[['petal_length']]
Y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

X = df[['petal_width']]
Y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

X = df[['sepal_length']]
Y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

X = df[['sepal_width']]
Y = df['species']

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_test)

import seaborn as sns
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"sepal_length").add_legend()
sns.FacetGrid(df,hue="species",height=3).map(sns.distplot,"sepal_width").add_legend()
plt.show()