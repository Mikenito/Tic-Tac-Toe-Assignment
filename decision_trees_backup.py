import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing


#import data set
data = pd.read_csv('ttt_data.csv', sep= ',')
data.rename(columns={
	'x': 'top-left-square', 
	'x.1': 'top-middle-square', 
	'x.2': 'top-right-square',
	'x.3': 'middle-left-square', 
	'o': 'middle-middle-square', 
	'o.1' : 'middle-right-square', 
	'x.4' : 'bottom-left-square', 
	'o.2' : 'bottom-middle-square', 
	'o.3':'bottom-right-square',
	'positive' : 'outcome'},
	inplace=True)


#print out data set and its information
print("Dataset Length:: ", len(data))
print('')
print("Dataset Shape:: ", data.shape)
print('')	
print("Dataset:: ", data.head(7))
print('')
print(data['outcome'].value_counts())
print('')

try:
	#data has to be processed since the predictor only takes in floats
	le = preprocessing.LabelEncoder()
	processed_data = data.apply(le.fit_transform)

except ValueError, e:
	print('processing error : ', e)
	

#define your X's and Y's	
X = processed_data.values[:, 0:9]
Y = processed_data.values[:,9]

#define X and Y training and testing data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.4, random_state = 100)

#print out training and testing data
print("X_train :: ", X_train)
print('')
print("Y_train :: ", y_train)
print('')
print("X_test :: ", X_test)
print('')
print("Y_test :: ", y_test)
print('')


try: 
	#use decision tree classifier to calculate entropy
	clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
	
except ValueError, e:
	print('clf_entropy did not run')
	print ('error : ', e)


try: 
	clf_entropy.fit(X_train, y_train)
	
	#y prediction with entropy
	y_pred_en = clf_entropy.predict(X_test)
	
	
	#print results
	print('this part ran')
	print('')
	print('y_pred_en :: ', y_pred_en)
	print('')
	print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100
	print('')
	
except ValueError, e:
	print('clf_entropy.fit did not run')
	print ('entropy error : ', e)



try:	
	#create a test query for the algorithm
	example_X = processed_data.values[469:470, 0:9]
	example_predict_y = clf_entropy.predict(example_X)
	print("for an example input :: ")
	print(example_X[0])
	print()
	print('The predicted outcome is :: ')
	if example_predict_y[0] == 0 :
		print('Negative')
		print()
	elif example_predict_y[0] == 1:
		print('Positive')
		print()
		 
except ValueError, e:
	print("This didn't work because : ", e)

