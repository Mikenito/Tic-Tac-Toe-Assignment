import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
# from sklearn.cross_validation import train_test_split
# from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score , classification_report
# from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
from sklearn.model_selection import cross_validate, train_test_split

csv_filename = 'test_csv2.csv'

df = pd.read_csv(csv_filename)
x = df.iloc[0:5,:]
df['Class'].unique() #does ['positive' 'negative']

le = preprocessing.LabelEncoder()
for col in df.columns:
	df[col] = le.fit_transform(df[col])

# for col in df.columns:				#returns 0's in the positions of x's and o's
# 	df[col] = pd.get_dummies(df[col]) 	#returns 1's in the positions of b's

features = (list(df.columns[:-1])) #does ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']

X = df[features]
Y = df['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
print("Naive Bayes")
nb = BernoulliNB()
clf_nb = nb.fit(X_train, Y_train)

print('Class: ', clf_nb.predict(X_test))
print('Accuracy: ', clf_nb.score(X_test, Y_test))
# print(X_train, Y_train) #(574, 9) (574,) tells us the number of test data - 574 data points
