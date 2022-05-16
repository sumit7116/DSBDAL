####Assignment 1######
import pandas as pd
df= pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
df

df.head()

df.tail()

df.info()

df.shape

df.describe()

df.isnull().sum()

df.isna().sum()

df.dtypes

df['species']=df['species'].astype('category')

df.dtypes

df['species'] = df['species'].cat.codes

df

####Assignment 2######
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


####Assignment 3######
import pandas as pd
df=pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
df

df.dtypes

df['species']=df['species'].astype('category')

df.dtypes

df['sepal_length'].groupby(df['species']).mean()

df['sepal_length'].groupby(df['species']).median()

df['sepal_length'].groupby(df['species']).min()

df['sepal_length'].groupby(df['species']).max()

df['sepal_length'].groupby(df['species']).std()

df[df['species']=='versicolor'].mean()