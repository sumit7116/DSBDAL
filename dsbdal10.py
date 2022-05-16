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