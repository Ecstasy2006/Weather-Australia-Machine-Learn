#Artificial Intelligence with Python | Artificial Intelligence Tutorial using Python | Edureka
#https://www.youtube.com/watch?v=7O60HOZRLng
import numpy as np
import pandas as pd

df = pd.read_csv('C://Users//Nigga//Documents//Python//weatherAUS.csv')
print('Size of weather data drame is: ',df.shape)

print(df[0:5])

print(df.count().sort_values())

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'], axis=1)

df = df.dropna(how='any')

print(df.shape)

from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))

print(z)

df = df[(z < 3).all(axis=1)]

print(df.shape)

df['RainToday'].replace({'No': 0, 'Yes':1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes':1},inplace = True)


categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(df[col]))

df = pd.get_dummies(df, columns=categorical_columns)
print(df.iloc[4:9])


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
print(df.iloc[4:10])

from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
selector=SelectKBest(chi2, k=3)
selector.fit(X,y)
X_new=selector.transform(X)
print(X.columns[selector.get_support(indices=True)])


df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
X = df[['Humidity3pm']]
Y = df[['RainTomorrow']]


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

#Calculating the accuracy
t0=time.time()
#Data S
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)
clf_dt = DecisionTreeClassifier(random_state=0)

#Building the model
clf_dt.fit(X_train,y_train)

#Evaluating the model
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test, y_pred)

#Printing
print('Accuracy using Decision Tree Classifier:',score)
print('Time taken using Decision Tree Classifier:',time.time()-t0)

'''

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Calculating the accuracy
t0=time.time()
#Data slicing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
#Building the model using the training data set
clf_logreg.fit(X_train, y_train)

#Evaluating the model using testing datea set
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test, y_pred)

#Printing the accuracy
print('Accuracy using Logistic Regression:',score)
print('Time take using Logistic Regression:',time.time()-t0)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Calculating the accuracy
t0=time.time()
#Data Splicing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)
clf_rf = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=0)
#Building the model
clf_rf.fit(X_train,y_train)
#Evaluating
y_pred=clf_rf.predict(X_test)
score=accuracy_score(y_test,y_pred)
#Pringint the accuracy and the time taken by classifier
print("Accuracy using Random Forest Classifier=",score)
print('Time taken using Random Forest Classifier=',time.time()-t0)

#Support Vector Machine
from sklearn import svm
from sklearn.model_selection import train_test_split

#Calculating
t0=time.time()
#Data S
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25)
clf_svc = svm.SVC(kernel='linear')

#Building the model
clf_svc.fit(X_train,y_train)

#Evaluating the model
y_pred = clf_svc.predict(X_test)
score = accuracy_score(y_test, y_pred)

#Printing
print('Accuracy using Support-vector machine:',score)
print('Time taken using Support-vector machine:',time.time()-t0)

'''








