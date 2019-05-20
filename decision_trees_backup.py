import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing


data = pd.read_csv('ttt_data.csv', sep= ',')

print("Dataset Length:: ", len(data))
print("Dataset Shape:: ", data.shape)

	
print("Dataset:: ", data.head(7), "\n\n")

try:
	le = preprocessing.LabelEncoder()
	data = data.apply(le.fit_transform)

except ValueError, e:
	print('processing error : ', e)
	
	
X = data.values[:, 1:10]
Y = data.values[:,9]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.4, random_state = 100)



print("X_train :: ", X_train)
print("Y_train :: ", y_train)
print("X_test :: ", X_test)
print("Y_test :: ", y_test)


try: 
	clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
	
except ValueError, e:
	print('clf_entropy did not run')
	print ('error : ', e)


try: 
	clf_entropy.fit(X_train, y_train)
	y_pred_en = clf_entropy.predict(X_test)
	#y_pred_en
	print('this part ran')
	print('y_pred-en :: ', y_pred_en)
	print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100
	
except ValueError, e:
	print('clf_entropy.fit did not run')
	print ('entropy error : ', e)

