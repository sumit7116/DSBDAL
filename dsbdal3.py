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