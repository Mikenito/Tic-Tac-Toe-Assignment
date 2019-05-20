# o is 0
# x is 1
# b is 2
# negative is 0
# positive is 1

import numpy as np
dataset = []
#reading the file and storing the data in a list
def readFileAndStoreDataset():
	f = open("tic-tac-toe.data")
	list = []
	list2 = []
	#read each line in tic-tac-toe.data and store it in a list
	for i in range(0,958):
	    list = f.readline()
	    #form a vector from individual elements in each line 
	    data = list.split(",")
	    #converts the data to 1's, 0's and 2's
	    for i in range(0,10):
	        #print(data[i])
	        list2.append(data[i])

	j = 0
	while j < 9580:
	    dataset.append(list2[j:j+10])
	    j+=10

	return dataset

#calculating prior probabilities for positive and negative class values
def calculatePriorProbabilities():
	totalDataPoints = 0
	posCount = 0
	negCount = 0
	for index in readFileAndStoreDataset():
		totalDataPoints +=1
		if index[-1] == 'positive\n':
			posCount +=1
		else:
			negCount +=1
	posPrior = posCount/totalDataPoints
	negPrior = negCount/totalDataPoints
	#is at: posCount = 2, negCount = 3
	return [posPrior, negPrior, posCount, negCount, totalDataPoints]

#cant use this because it changes the pxcount,pocount,and pbcount values
# list = calculatePriorProbabilities()

#calculating Conditional probabilities for each attribute value
def calcCondProbPos():
	attrA = [['A1x',0],['A1o',0],['A1b',0],
			 ['A2x',0],['A2o',0],['A2b',0],
			 ['A3x',0],['A3o',0],['A3b',0],
			 ['A4x',0],['A4o',0],['A4b',0],
			 ['A5x',0],['A5o',0],['A5b',0],
			 ['A6x',0],['A6o',0],['A6b',0],
			 ['A7x',0],['A7o',0],['A7b',0],
			 ['A8x',0],['A8o',0],['A8b',0],
			 ['A9x',0],['A9o',0],['A9b',0]]
	a1pxcount = 0
	a1pocount = 0
	a1pbcount = 0
	a2pxcount = 0
	a2pocount = 0
	a2pbcount = 0
	a3pxcount = 0
	a3pocount = 0
	a3pbcount = 0
	a4pxcount = 0
	a4pocount = 0
	a4pbcount = 0
	a5pxcount = 0
	a5pocount = 0
	a5pbcount = 0
	a6pxcount = 0
	a6pocount = 0
	a6pbcount = 0
	a7pxcount = 0
	a7pocount = 0
	a7pbcount = 0
	a8pxcount = 0
	a8pocount = 0
	a8pbcount = 0
	a9pxcount = 0
	a9pocount = 0
	a9pbcount = 0
	for index in readFileAndStoreDataset():
		if index[0] == 'x' and index[-1] == 'positive\n':
			a1pxcount+=1
			attrA[0][1] = a1pxcount/626
		if index[0] == 'o' and index[-1] == 'positive\n':
			a1pocount+=1
			attrA[1][1] = a1pocount/626
		if index[0] == 'b' and index[-1] == 'positive\n':
			a1pbcount+=1
			attrA[2][1] = a1pbcount/626
		if index[1] == 'x' and index[-1] == 'positive\n':
			a2pxcount+=1
			attrA[3][1] = a2pxcount/626
		if index[1] == 'o' and index[-1] == 'positive\n':
			a2pocount+=1
			attrA[4][1] = a2pocount/626
		if index[1] == 'b' and index[-1] == 'positive\n':
			a2pbcount+=1
			attrA[5][1] = a2pbcount/626
		if index[2] == 'x' and index[-1] == 'positive\n':
			a3pxcount+=1
			attrA[6][1] = a3pxcount/626
		if index[2] == 'o' and index[-1] == 'positive\n':
			a3pocount+=1
			attrA[7][1] = a3pocount/626
		if index[2] == 'b' and index[-1] == 'positive\n':
			a3pbcount+=1
			attrA[8][1] = a3pbcount/626
		if index[3] == 'x' and index[-1] == 'positive\n':
			a4pxcount+=1
			attrA[9][1] = a4pxcount/626
		if index[3] == 'o' and index[-1] == 'positive\n':
			a4pocount+=1
			attrA[10][1] = a4pocount/626
		if index[3] == 'b' and index[-1] == 'positive\n':
			a4pbcount+=1
			attrA[11][1] = a4pbcount/626
		if index[4] == 'x' and index[-1] == 'positive\n':
			a5pxcount+=1
			attrA[12][1] = a5pxcount/626
		if index[4] == 'o' and index[-1] == 'positive\n':
			a5pocount+=1
			attrA[13][1] = a5pocount/626
		if index[4] == 'b' and index[-1] == 'positive\n':
			a5pbcount+=1
			attrA[14][1] = a5pbcount/626
		if index[5] == 'x' and index[-1] == 'positive\n':
			a6pxcount+=1
			attrA[15][1] = a6pxcount/626
		if index[5] == 'o' and index[-1] == 'positive\n':
			a6pocount+=1
			attrA[16][1] = a6pocount/626
		if index[5] == 'b' and index[-1] == 'positive\n':
			a6pbcount+=1
			attrA[17][1] = a6pbcount/626
		if index[6] == 'x' and index[-1] == 'positive\n':
			a7pxcount+=1
			attrA[18][1] = a7pxcount/626
		if index[6] == 'o' and index[-1] == 'positive\n':
			a7pocount+=1
			attrA[19][1] = a7pocount/626
		if index[6] == 'b' and index[-1] == 'positive\n':
			a7pbcount+=1
			attrA[20][1] = a7pbcount/626
		if index[7] == 'x' and index[-1] == 'positive\n':
			a8pxcount+=1
			attrA[21][1] = a8pxcount/626
		if index[7] == 'o' and index[-1] == 'positive\n':
			a8pocount+=1
			attrA[22][1] = a8pocount/626
		if index[7] == 'b' and index[-1] == 'positive\n':
			a8pbcount+=1
			attrA[23][1] = a8pbcount/626
		if index[8] == 'x' and index[-1] == 'positive\n':
			a9pxcount+=1
			attrA[24][1] = a9pxcount/626
		if index[8] == 'o' and index[-1] == 'positive\n':
			a9pocount+=1
			attrA[25][1] = a9pocount/626
		if index[8] == 'b' and index[-1] == 'positive\n':
			a9pbcount+=1
			attrA[26][1] = a9pbcount/626
	# attrA1[0][1] = pxcount/(calculatePriorProbabilities()[2]/2)
	# attrA1[1][1] = pocount/(calculatePriorProbabilities()[2]/3)
	# attrA1[2][1] = pbcount/(calculatePriorProbabilities()[2]/4)	
	for i in attrA:
		print(i)

def calcCondProbNeg():
	nattrA = [['A1x',0],['A1o',0],['A1b',0],
			 ['A2x',0],['A2o',0],['A2b',0],
			 ['A3x',0],['A3o',0],['A3b',0],
			 ['A4x',0],['A4o',0],['A4b',0],
			 ['A5x',0],['A5o',0],['A5b',0],
			 ['A6x',0],['A6o',0],['A6b',0],
			 ['A7x',0],['A7o',0],['A7b',0],
			 ['A8x',0],['A8o',0],['A8b',0],
			 ['A9x',0],['A9o',0],['A9b',0]]
	a1nxcount = 0
	a1nocount = 0
	a1nbcount = 0
	a2nxcount = 0
	a2nocount = 0
	a2nbcount = 0
	a3nxcount = 0
	a3nocount = 0
	a3nbcount = 0
	a4nxcount = 0
	a4nocount = 0
	a4nbcount = 0
	a5nxcount = 0
	a5nocount = 0
	a5nbcount = 0
	a6nxcount = 0
	a6nocount = 0
	a6nbcount = 0
	a7nxcount = 0
	a7nocount = 0
	a7nbcount = 0
	a8nxcount = 0
	a8nocount = 0
	a8nbcount = 0
	a9nxcount = 0
	a9nocount = 0
	a9nbcount = 0
	for index in readFileAndStoreDataset():
		if index[0] == 'x' and index[-1] == 'negative\n':
			a1nxcount+=1
			nattrA[0][1] = a1nxcount/332
		if index[0] == 'o' and index[-1] == 'negative\n':
			a1nocount+=1
			nattrA[1][1] = a1nocount/332
		if index[0] == 'b' and index[-1] == 'negative\n':
			a1nbcount+=1
			nattrA[2][1] = a1nbcount/332
		if index[1] == 'x' and index[-1] == 'negative\n':
			a2nxcount+=1
			nattrA[3][1] = a2nxcount/332
		if index[1] == 'o' and index[-1] == 'negative\n':
			a2nocount+=1
			nattrA[4][1] = a2nocount/332
		if index[1] == 'b' and index[-1] == 'negative\n':
			a2nbcount+=1
			nattrA[5][1] = a2nbcount/332
		if index[2] == 'x' and index[-1] == 'negative\n':
			a3nxcount+=1
			nattrA[6][1] = a3nxcount/332
		if index[2] == 'o' and index[-1] == 'negative\n':
			a3nocount+=1
			nattrA[7][1] = a3nocount/332
		if index[2] == 'b' and index[-1] == 'negative\n':
			a3nbcount+=1
			nattrA[8][1] = a3nbcount/332
		if index[3] == 'x' and index[-1] == 'negative\n':
			a4nxcount+=1
			nattrA[9][1] = a4nxcount/332
		if index[3] == 'o' and index[-1] == 'negative\n':
			a4nocount+=1
			nattrA[10][1] = a4nocount/332
		if index[3] == 'b' and index[-1] == 'negative\n':
			a4nbcount+=1
			nattrA[11][1] = a4nbcount/332
		if index[4] == 'x' and index[-1] == 'negative\n':
			a5nxcount+=1
			nattrA[12][1] = a5nxcount/332
		if index[4] == 'o' and index[-1] == 'negative\n':
			a5nocount+=1
			nattrA[13][1] = a5nocount/332
		if index[4] == 'b' and index[-1] == 'negative\n':
			a5nbcount+=1
			nattrA[14][1] = a5nbcount/332
		if index[5] == 'x' and index[-1] == 'negative\n':
			a6nxcount+=1
			nattrA[15][1] = a6nxcount/332
		if index[5] == 'o' and index[-1] == 'negative\n':
			a6nocount+=1
			nattrA[16][1] = a6nocount/332
		if index[5] == 'b' and index[-1] == 'negative\n':
			a6nbcount+=1
			nattrA[17][1] = a6nbcount/332
		if index[6] == 'x' and index[-1] == 'negative\n':
			a7nxcount+=1
			nattrA[18][1] = a7nxcount/332
		if index[6] == 'o' and index[-1] == 'negative\n':
			a7nocount+=1
			nattrA[19][1] = a7nocount/332
		if index[6] == 'b' and index[-1] == 'negative\n':
			a7nbcount+=1
			nattrA[20][1] = a7nbcount/332
		if index[7] == 'x' and index[-1] == 'negative\n':
			a8nxcount+=1
			nattrA[21][1] = a8nxcount/332
		if index[7] == 'o' and index[-1] == 'negative\n':
			a8nocount+=1
			nattrA[22][1] = a8nocount/332
		if index[7] == 'b' and index[-1] == 'negative\n':
			a8nbcount+=1
			nattrA[23][1] = a8nbcount/332
		if index[8] == 'x' and index[-1] == 'negative\n':
			a9nxcount+=1
			nattrA[24][1] = a9nxcount/332
		if index[8] == 'o' and index[-1] == 'negative\n':
			a9nocount+=1
			nattrA[25][1] = a9nocount/332
		if index[8] == 'b' and index[-1] == 'negative\n':
			a9nbcount+=1
			nattrA[26][1] = a9nbcount/332	
	for i in nattrA:
		print(i)

# calcCondProbNeg()
calcCondProbPos()
# print(readFileAndStoreDataset()[-1])