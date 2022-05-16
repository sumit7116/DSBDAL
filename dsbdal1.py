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