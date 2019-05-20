import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from time import time
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
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

features = (list(df.columns[:])) #does ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']

X = df[features]
Y = df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# print(X_train[0:5])
# print(Y_train[0:5])
print("Length of features", len(features)) #length of features
print("Length of X_train dataset", len(X_train))
print("Length of X_test dataset", len(X_test))
print('Training Data')
print(X_train[0:5].loc[:,"A1":"Class"]) #prints attribute values
# print(Y_train[0:5]) #Prints the class values
# print('Element at position row,col :',X_train.iat[0, 8]) #for testing

#Calculate Conditional Probabililities 
def calProbs():
	posCount, negCount = 0, 0
	posPriorProb, negPriorPorb = 0,0
	posxCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	posoCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	posbCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	negxCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	negoCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	negbCondProbs = [['A1',0],['A2',0],['A3',0],['A4',0],['A5',0],['A6',0],['A7',0],['A8',0],['A9',0]]
	#calculates number of positives and negatives
	for i in Y_train:
		if i == 1: posCount+=1
		else: negCount +=1
	# print("Positive Count: ",posCount,"Negative Count: ",negCount)
	#count number of (x,o,b) such that (positive/negative)
	posPriorProb = posCount/(posCount+negCount)
	negPriorPorb = negCount/(posCount+negCount)
	# print(posPriorProb, negPriorPorb)
	# print(posCount, negCount)
	z=0
	while z < (len(features)-1):
		negxcount, posxcount = 0,0
		negocount, posocount = 0,0
		negbcount, posbcount = 0,0
		for x in range(0,len(X_train)):
			if X_train.iat[x,z] == 2 and Y_train.iat[x] == 1:
				posxcount+=1
			elif X_train.iat[x,z] == 2 and Y_train.iat[x] == 0:
				negxcount+=1
			elif X_train.iat[x,z] == 1 and Y_train.iat[x]== 1:
				posocount+=1
			elif X_train.iat[x,z] == 1 and Y_train.iat[x] == 0:
				negocount+=1
			elif X_train.iat[x,z] == 0 and Y_train.iat[x]== 1:
				posbcount+=1
			elif X_train.iat[x,z] == 0 and Y_train.iat[x] == 0:
				negbcount+=1
			# print(X_train.iat[x,z], Y_train.iat[x])
		posxCondProbs[z][1] = posxcount/posCount
		posoCondProbs[z][1] = posocount/posCount
		posbCondProbs[z][1] = posbcount/posCount
		negxCondProbs[z][1] = negxcount/negCount
		negoCondProbs[z][1] = negocount/negCount
		negbCondProbs[z][1] = negbcount/negCount
		# print("Neg x count for each attribute: ",negxcount/negCount) #this is for testing
		z+=1
	# print("x|pos Conditional Probabilities: ",posxCondProbs)
	# print("o|pos Conditional Probabilities: ",posoCondProbs)
	# print("b|pos Conditional Probabilities: ",posbCondProbs)
	# print("x|neg Conditional Probabilities: ",negxCondProbs)
	# print("o|neg Conditional Probabilities: ",negoCondProbs)
	# print("b|neg Conditional Probabilities: ",negbCondProbs)
	return [posxCondProbs,posoCondProbs,posbCondProbs, negxCondProbs,negoCondProbs, negbCondProbs]

print("Testing Data")
print(X_test.loc[:,'A1':'Class'])
#storing conditional probabilities in global arrays
posXCondProb = calProbs()[0]
posOCondProb = calProbs()[1]
posBCondProb = calProbs()[2]
negXCondProb = calProbs()[3]
negOCondProb = calProbs()[4]
negBCondProb = calProbs()[5]

def predictClass(X_train):
	y,z = 0,0
	while z < len(X_test) and y<len(Y_test):
		posProduct = 1
		negProduct = 1
		for i in range(0,(len(features)-1)):
			if X_test.iat[z,i] == 2 and Y_test.iat[y] == 1:
				# print(posXCondProb[i][1])
				posProduct *= posXCondProb[i][1]
			elif X_test.iat[z,i] == 1 and Y_test.iat[y] == 1:
				# print(posOCondProb[i][1])
				posProduct *= posOCondProb[i][1]
			elif X_test.iat[z,i] == 0 and Y_test.iat[y] == 1:
				# print(posBCondProb[i][1])
				posProduct *= posBCondProb[i][1]
			elif X_test.iat[z,i] == 2 and Y_test.iat[y] == 0:
				# print(negXCondProb[i][1])
				negProduct*= negXCondProb[i][1]
			elif X_test.iat[z,i] == 1 and Y_test.iat[y] == 0:
				# print(negOCondProb[i][1])
				negProduct*= negOCondProb[i][1]
			elif X_test.iat[z,i] == 0 and Y_test.iat[y] == 0:
				# print(negBCondProb[i][1])
				negProduct*= negBCondProb[i][1]
		print("Line number: ", z)
		if posProduct > negProduct:
			print("Lose for X")
		else: print("Win for X")
		# print("Product of positive A|positive",posProduct)
		# print("Product of negative A|negative",negProduct)
			# print(posProduct)
			# print(X_test.iat[z,i], Y_test.iat[y])
		z+=1
		y+=1

predictClass(X_test)
# print("Naive Bayes")
# nb = BernoulliNB()
# clf_nb = nb.fit(X_train, Y_train)

# print('Class: ', clf_nb.predict(X_test))
# print('Accuracy: ', clf_nb.score(X_test, Y_test))
# print(X_train, Y_train) #(574, 9) (574,) tells us the number of test data - 574 data points
