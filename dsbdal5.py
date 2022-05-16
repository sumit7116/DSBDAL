import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv")
df

df.shape

df.isna().sum()

import seaborn as sns
sns.heatmap(df.corr(), annot=True)

sns.boxplot(df['EstimatedSalary'])

df['Gender']=df['Gender'].astype('category')
df['Gender'] = df['Gender'].cat.codes

df.dtypes

df['Gender'].value_counts()

from sklearn.model_selection import train_test_split
x = df[['Age','EstimatedSalary']]
y = df['Purchased']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression()
m1.fit(x_train,y_train)
y_pred = m1.predict(x_test)
x_test.head()

y_test.head()

print("Model Score :",m1.score(x_test,y_test)*100)

from sklearn.metrics import mean_absolute_error

print("MAE :",mean_absolute_error(y_test,y_pred))

from sklearn.preprocessing import MinMaxScaler
#fit Scaler on training data
norm=MinMaxScaler().fit(x_train)
#transform training data
x_train=norm.transform(x_train)
#fit Scaler on testing data
norm=MinMaxScaler().fit(x_test)
#transform testing data
x_test=norm.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)
print ('Model Score:',model.score(x_test,y_test))

model.predict(x_test)

print("Model Score :",model.score(x_test,y_test)*100)

from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(y_test,y_pred)#actual o/p and predicted output
print(cf_matrix)

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y_test,y_pred,average='macro'))

print(precision_recall_fscore_support(y_test,y_pred,average='micro'))

print(precision_recall_fscore_support(y_test,y_pred,average='weighted'))

from sklearn.metrics import precision_recall_fscore_support
score=precision_recall_fscore_support(y_test,y_pred,average='micro')
print('Precision of Model:',score[0])
print('Recall of Model:',score[1])
print('F-score of Model:',score[2])