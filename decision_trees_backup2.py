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
print(data['top-left-square'].value_counts())
print('')
print(data['top-middle-square'].value_counts())
print('')
print(data['top-right-square'].value_counts())
print('')
print(data['middle-left-square'].value_counts())
print('')
print(data['middle-middle-square'].value_counts())
print('')
print(data['middle-right-square'].value_counts())
print('')
print(data['bottom-left-square'].value_counts())
print('')
print(data['bottom-middle-square'].value_counts())
print('')
print(data['bottom-right-square'].value_counts())
print('')
	
	

pos_prob = float(625)/float(957)	
	
neg_prob = float(332)/float(957)	

def entropy_targetClass():
    '''
    return the Entropy of the target class of outcomes:
    '''
    return float(0) - (pos_prob*np.log2(pos_prob) + neg_prob*np.log2(neg_prob))




try:
	#data has to be processed since the predictor only takes in floats
	le = preprocessing.LabelEncoder()
	processed_data = data.apply(le.fit_transform)

except ValueError, e:
	print('processing error : ', e)
	

#define your X's and Y's	
X = processed_data.values[:, 0:9]
Y = processed_data.values[:,9]


top_left_posX_count = 0
top_left_negX_count = 0
top_left_posO_count = 0
top_left_negO_count = 0
top_left_posB_count = 0
top_left_negB_count = 0

top_middle_posX_count = 0
top_middle_negX_count = 0
top_middle_posO_count = 0
top_middle_negO_count = 0
top_middle_posB_count = 0
top_middle_negB_count = 0

top_right_posX_count = 0
top_right_negX_count = 0
top_right_posO_count = 0
top_right_negO_count = 0
top_right_posB_count = 0
top_right_negB_count = 0

middle_left_posX_count = 0
middle_left_negX_count = 0
middle_left_posO_count = 0
middle_left_negO_count = 0
middle_left_posB_count = 0
middle_left_negB_count = 0

middle_middle_posX_count = 0
middle_middle_negX_count = 0
middle_middle_posO_count = 0
middle_middle_negO_count = 0
middle_middle_posB_count = 0
middle_middle_negB_count = 0

middle_right_posX_count = 0
middle_right_negX_count = 0
middle_right_posO_count = 0
middle_right_negO_count = 0
middle_right_posB_count = 0
middle_right_negB_count = 0

bottom_left_posX_count = 0
bottom_left_negX_count = 0
bottom_left_posO_count = 0
bottom_left_negO_count = 0
bottom_left_posB_count = 0
bottom_left_negB_count = 0

bottom_middle_posX_count = 0
bottom_middle_negX_count = 0
bottom_middle_posO_count = 0
bottom_middle_negO_count = 0
bottom_middle_posB_count = 0
bottom_middle_negB_count = 0

bottom_right_posX_count = 0
bottom_right_negX_count = 0
bottom_right_posO_count = 0
bottom_right_negO_count = 0
bottom_right_posB_count = 0
bottom_right_negB_count = 0





top_left = data.values[:, 0]
top_middle = data.values[:, 1]
top_right = data.values[:, 2]

middle_left = data.values[:, 3]
middle_middle = data.values[:, 4]
middle_right = data.values[:, 5]

bottom_left = data.values[:, 6]
bottom_middle = data.values[:, 7]
bottom_right = data.values[:, 8]

outcomes = data.values[:, 9]

iterator = 0


while iterator < 957:
	
	#The X's
	#1
	if top_left[iterator] == 'x' and outcomes[iterator] == 'positive':
			top_left_posX_count += 1
	elif top_left[iterator] == 'x' and outcomes[iterator] == 'negative': 
			top_left_negX_count += 1
	
	
	#2		
	if top_middle[iterator] == 'x' and outcomes[iterator] == 'positive':
			top_middle_posX_count += 1
	elif top_middle[iterator] == 'x' and outcomes[iterator] == 'negative':
			top_middle_negX_count += 1			
	
	
	#3
	if top_right[iterator] == 'x' and outcomes[iterator] == 'positive':
			top_right_posX_count += 1
	elif top_right[iterator] == 'x' and outcomes[iterator] == 'negative':
			top_right_negX_count += 1		
	
						
	#4
	if middle_left[iterator] == 'x' and outcomes[iterator] == 'positive':
			middle_left_posX_count += 1
	elif middle_left[iterator] == 'x' and outcomes[iterator] == 'negative':
			middle_left_negX_count += 1		

	
	#5
	if middle_middle[iterator] == 'x' and outcomes[iterator] == 'positive':
			middle_middle_posX_count += 1
	elif middle_middle[iterator] == 'x' and outcomes[iterator] == 'negative':
			middle_middle_negX_count += 1
	
	
	#6		
	if middle_right[iterator] == 'x' and outcomes[iterator] == 'positive':
			middle_right_posX_count += 1
	elif middle_right[iterator] == 'x' and outcomes[iterator] == 'negative':
			middle_right_negX_count += 1
	
	
	#7		
	if bottom_left[iterator] == 'x' and outcomes[iterator] == 'positive':
			bottom_left_posX_count += 1
	elif bottom_left[iterator] == 'x' and outcomes[iterator] == 'negative':
			bottom_left_negX_count += 1
	
	
	#8		
	if bottom_middle[iterator] == 'x' and outcomes[iterator] == 'positive':
			bottom_middle_posX_count += 1
	elif bottom_middle[iterator] == 'x' and outcomes[iterator] == 'negative':
			bottom_middle_negX_count += 1
			
	#9
	if bottom_right[iterator] == 'x' and outcomes[iterator] == 'positive':
			bottom_right_posX_count += 1
	elif bottom_right[iterator] == 'x' and outcomes[iterator] == 'negative':
			bottom_right_negX_count += 1	
	
	
	#The O's
	#1
	if top_left[iterator] == 'o' and outcomes[iterator] == 'positive':
			top_left_posO_count += 1
	elif top_left[iterator] == 'o' and outcomes[iterator] == 'negative': 
			top_left_negO_count += 1
	
	
	#2		
	if top_middle[iterator] == 'o' and outcomes[iterator] == 'positive':
			top_middle_posO_count += 1
	elif top_middle[iterator] == 'o' and outcomes[iterator] == 'negative':
			top_middle_negO_count += 1			
	
	
	#3
	if top_right[iterator] == 'o' and outcomes[iterator] == 'positive':
			top_right_posO_count += 1
	elif top_right[iterator] == 'o' and outcomes[iterator] == 'negative':
			top_right_negO_count += 1		
	
						
	#4
	if middle_left[iterator] == 'o' and outcomes[iterator] == 'positive':
			middle_left_posO_count += 1
	elif middle_left[iterator] == 'o' and outcomes[iterator] == 'negative':
			middle_left_negO_count += 1		

	
	#5
	if middle_middle[iterator] == 'o' and outcomes[iterator] == 'positive':
			middle_middle_posO_count += 1
	elif middle_middle[iterator] == 'o' and outcomes[iterator] == 'negative':
			middle_middle_negO_count += 1
	
	
	#6		
	if middle_right[iterator] == 'o' and outcomes[iterator] == 'positive':
			middle_right_posO_count += 1
	elif middle_right[iterator] == 'o' and outcomes[iterator] == 'negative':
			middle_right_negO_count += 1
	
	
	#7		
	if bottom_left[iterator] == 'o' and outcomes[iterator] == 'positive':
			bottom_left_posO_count += 1
	elif bottom_left[iterator] == 'o' and outcomes[iterator] == 'negative':
			bottom_left_negO_count += 1
	
	
	#8		
	if bottom_middle[iterator] == 'o' and outcomes[iterator] == 'positive':
			bottom_middle_posO_count += 1
	elif bottom_middle[iterator] == 'o' and outcomes[iterator] == 'negative':
			bottom_middle_negO_count += 1
			
	#9
	if bottom_right[iterator] == 'o' and outcomes[iterator] == 'positive':
			bottom_right_posO_count += 1
	elif bottom_right[iterator] == 'o' and outcomes[iterator] == 'negative':
			bottom_right_negO_count += 1	
		
	
	#The B's
	#1
	if top_left[iterator] == 'b' and outcomes[iterator] == 'positive':
			top_left_posB_count += 1
	elif top_left[iterator] == 'b' and outcomes[iterator] == 'negative': 
			top_left_negB_count += 1
	
	
	#2		
	if top_middle[iterator] == 'b' and outcomes[iterator] == 'positive':
			top_middle_posB_count += 1
	elif top_middle[iterator] == 'b' and outcomes[iterator] == 'negative':
			top_middle_negB_count += 1			
	
	
	#3
	if top_right[iterator] == 'b' and outcomes[iterator] == 'positive':
			top_right_posB_count += 1
	elif top_right[iterator] == 'b' and outcomes[iterator] == 'negative':
			top_right_negB_count += 1		
	
						
	#4
	if middle_left[iterator] == 'b' and outcomes[iterator] == 'positive':
			middle_left_posB_count += 1
	elif middle_left[iterator] == 'b' and outcomes[iterator] == 'negative':
			middle_left_negB_count += 1		

	
	#5
	if middle_middle[iterator] == 'b' and outcomes[iterator] == 'positive':
			middle_middle_posB_count += 1
	elif middle_middle[iterator] == 'b' and outcomes[iterator] == 'negative':
			middle_middle_negB_count += 1
	
	
	#6		
	if middle_right[iterator] == 'b' and outcomes[iterator] == 'positive':
			middle_right_posB_count += 1
	elif middle_right[iterator] == 'b' and outcomes[iterator] == 'negative':
			middle_right_negB_count += 1
	
	
	#7		
	if bottom_left[iterator] == 'b' and outcomes[iterator] == 'positive':
			bottom_left_posB_count += 1
	elif bottom_left[iterator] == 'b' and outcomes[iterator] == 'negative':
			bottom_left_negB_count += 1
	
	
	#8		
	if bottom_middle[iterator] == 'b' and outcomes[iterator] == 'positive':
			bottom_middle_posB_count += 1
	elif bottom_middle[iterator] == 'b' and outcomes[iterator] == 'negative':
			bottom_middle_negB_count += 1
			
	#9
	if bottom_right[iterator] == 'b' and outcomes[iterator] == 'positive':
			bottom_right_posB_count += 1
	elif bottom_right[iterator] == 'b' and outcomes[iterator] == 'negative':
			bottom_right_negB_count += 1	
	
	
			
	iterator += 1									
		

#Print top values
print('top left posX count :: ' , top_left_posX_count)
print('top left negX count :: ' , top_left_negX_count)
print( 'top middle posX count :: ' , top_middle_posO_count)
print( 'top middle negX count :: ' , top_middle_negO_count)
print( 'top right posX count :: ' , top_right_posB_count)
print( 'top right negX count :: ' , top_right_negB_count)
print('')
print('top left posO count :: ' , top_left_posX_count)
print('top left negO count :: ' , top_left_negX_count)
print( 'top middle posO count :: ' , top_middle_posO_count)
print( 'top middle negO count :: ' , top_middle_negO_count)
print( 'top right posO count :: ' , top_right_posB_count)
print( 'top right negO count :: ' , top_right_negB_count)
print('')
print('top left posB count :: ' , top_left_posX_count)
print('top left negB count :: ' , top_left_negX_count)
print( 'top middle posB count :: ' , top_middle_posO_count)
print( 'top middle negB count :: ' , top_middle_negO_count)
print( 'top right posB count :: ' , top_right_posB_count)
print( 'top right negB count :: ' , top_right_negB_count)
print('')


#Print middle values
print( 'middle left posX count :: ' , middle_left_posX_count)
print( 'middle left negX count :: ' , middle_left_negX_count)
print( 'middle middle posX count :: ' , middle_middle_posO_count)
print( 'middle middle negX count :: ' , middle_middle_negO_count)
print( 'middle right posX count :: ' , middle_right_posB_count)
print( 'middle right negX count :: ' , middle_right_negB_count)
print('')
print( 'middle left posO count :: ' , middle_left_posX_count)
print( 'middle left negO count :: ' , middle_left_negX_count)
print( 'middle middle posO count :: ' , middle_middle_posO_count)
print( 'middle middle negO count :: ' , middle_middle_negO_count)
print( 'middle right posO count :: ' , middle_right_posB_count)
print( 'middle right negO count :: ' , middle_right_negB_count)
print('')
print( 'middle left posB count :: ' , middle_left_posX_count)
print( 'middle left negB count :: ' , middle_left_negX_count)
print( 'middle middle posB count :: ' , middle_middle_posO_count)
print( 'middle middle negB count :: ' , middle_middle_negO_count)
print( 'middle right posB count :: ' , middle_right_posB_count)
print( 'middle right negB count :: ' , middle_right_negB_count)
print('')


#Print bottom values
print( 'bottom left posX count :: ' , bottom_left_posX_count)
print( 'bottom left negX count :: ' , bottom_left_negX_count)
print( 'bottom middle posX count :: ' , bottom_middle_posO_count)
print( 'bottom middle negX count :: ' , bottom_middle_negO_count)
print( 'bottom right posX count :: ' , bottom_right_posB_count)
print( 'bottom right negX count :: ' , bottom_right_negB_count)
print('')
print( 'bottom left posO count :: ' , bottom_left_posX_count)
print( 'bottom left negO count :: ' , bottom_left_negX_count)
print( 'bottom middle posO count :: ' , bottom_middle_posO_count)
print( 'bottom middle negO count :: ' , bottom_middle_negO_count)
print( 'bottom right posO count :: ' , bottom_right_posB_count)
print( 'bottom right negO count :: ' , bottom_right_negB_count)
print('')
print( 'bottom left posB count :: ' , bottom_left_posX_count)
print( 'bottom left negB count :: ' , bottom_left_negX_count)
print( 'bottom middle posB count :: ' , bottom_middle_posO_count)
print( 'bottom middle negB count :: ' , bottom_middle_negO_count)
print( 'bottom right posB count :: ' , bottom_right_posB_count)
print( 'bottom right negB count :: ' , bottom_right_negB_count)
print('')


'''
Gain(D, top_left) where features are x, o and b

def gain_top_left():
	#for F = x
	#|D| = 417
	#H_x = -{negX/417*log2(negX/417) + top_left_posX_count/417*log2(posX/417)}
	
	#for F = o
	#|D| = 335
	#H_o = -{negO/335*log2(negO/335) + posO/335*log2(posO/335)}
	
	
	#for F = b
	#|D| = 205
	#H_b = -{negB/335*log2(negB/335) + posB/335*log2(posB/335)}
	
	gain = entropy_target_class() - 1/957{417*(H_x) + 335*(H_o) + 205*(H_b)}
	
	return gain
	
	##follow model for the rest of the gains
	
'''

def gain_top_left():
	h_x = float(0) - ( (float(top_left_posX_count)/float(417) )*np.log2(float(top_left_posX_count)/float(417)) + (float(top_left_negX_count)/float(417))*np.log2(float(top_left_negX_count)/float(417)) )
	h_o = float(0) - ( (float(top_left_posO_count)/float(335) )*np.log2(float(top_left_posO_count)/float(335)) + (float(top_left_negX_count)/float(335))*np.log2(float(top_left_negX_count)/float(335)) )
	h_b = float(0) - ( (float(top_left_posB_count)/float(205) )*np.log2(float(top_left_posB_count)/float(205)) + (float(top_left_negX_count)/float(205))*np.log2(float(top_left_negX_count)/float(205)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain

def gain_top_right():
	h_x = float(0) - ( (float(top_right_posX_count)/float(417) )*np.log2(float(top_right_posX_count)/float(417)) + (float(top_right_negX_count)/float(417))*np.log2(float(top_right_negX_count)/float(417)) )
	h_o = float(0) - ( (float(top_right_posO_count)/float(335) )*np.log2(float(top_right_posO_count)/float(335)) + (float(top_right_negX_count)/float(335))*np.log2(float(top_right_negX_count)/float(335)) )
	h_b = float(0) - ( (float(top_right_posB_count)/float(205) )*np.log2(float(top_right_posB_count)/float(205)) + (float(top_right_negX_count)/float(205))*np.log2(float(top_right_negX_count)/float(205)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain
	
def gain_top_middle():
	h_x = float(0) - ( (float(top_middle_posX_count)/float(337) )*np.log2(float(top_middle_posX_count)/float(337)) + (float(top_middle_negX_count)/float(337))*np.log2(float(top_middle_negX_count)/float(337)) )
	h_o = float(0) - ( (float(top_middle_posO_count)/float(330) )*np.log2(float(top_middle_posO_count)/float(330)) + (float(top_middle_negX_count)/float(330))*np.log2(float(top_middle_negX_count)/float(330)) )
	h_b = float(0) - ( (float(top_middle_posB_count)/float(250) )*np.log2(float(top_middle_posB_count)/float(250)) + (float(top_middle_negX_count)/float(250))*np.log2(float(top_middle_negX_count)/float(250)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain	


def gain_middle_left():
	h_x = float(0) - ( (float(middle_left_posX_count)/float(377) )*np.log2(float(middle_left_posX_count)/float(377)) + (float(middle_left_negX_count)/float(377))*np.log2(float(middle_left_negX_count)/float(377)) )
	h_o = float(0) - ( (float(middle_left_posO_count)/float(330) )*np.log2(float(middle_left_posO_count)/float(330)) + (float(middle_left_negX_count)/float(330))*np.log2(float(middle_left_negX_count)/float(330)) )
	h_b = float(0) - ( (float(middle_left_posB_count)/float(250) )*np.log2(float(middle_left_posB_count)/float(250)) + (float(middle_left_negX_count)/float(250))*np.log2(float(middle_left_negX_count)/float(250)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain

def gain_middle_right():
	h_x = float(0) - ( (float(middle_right_posX_count)/float(378) )*np.log2(float(middle_right_posX_count)/float(378)) + (float(middle_right_negX_count)/float(378))*np.log2(float(middle_right_negX_count)/float(378)) )
	h_o = float(0) - ( (float(middle_right_posO_count)/float(329) )*np.log2(float(middle_right_posO_count)/float(329)) + (float(middle_right_negX_count)/float(329))*np.log2(float(middle_right_negX_count)/float(329)) )
	h_b = float(0) - ( (float(middle_right_posB_count)/float(250) )*np.log2(float(middle_right_posB_count)/float(250)) + (float(middle_right_negX_count)/float(250))*np.log2(float(middle_right_negX_count)/float(250)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain
	
def gain_middle_middle():
	h_x = float(0) - ( (float(middle_middle_posX_count)/float(458) )*np.log2(float(middle_middle_posX_count)/float(458)) + (float(middle_middle_negX_count)/float(458))*np.log2(float(middle_middle_negX_count)/float(458)) )
	h_o = float(0) - ( (float(middle_middle_posO_count)/float(338) )*np.log2(float(middle_middle_posO_count)/float(338)) + (float(middle_middle_negX_count)/float(338))*np.log2(float(middle_middle_negX_count)/float(338)) )
	h_b = float(0) - ( (float(middle_middle_posB_count)/float(161) )*np.log2(float(middle_middle_posB_count)/float(161)) + (float(middle_middle_negX_count)/float(161))*np.log2(float(middle_middle_negX_count)/float(161)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain


def gain_bottom_left():
	h_x = float(0) - ( (float(middle_left_posX_count)/float(417) )*np.log2(float(middle_left_posX_count)/float(417)) + (float(middle_left_negX_count)/float(417))*np.log2(float(middle_left_negX_count)/float(417)) )
	h_o = float(0) - ( (float(middle_left_posO_count)/float(335) )*np.log2(float(middle_left_posO_count)/float(335)) + (float(middle_left_negX_count)/float(335))*np.log2(float(middle_left_negX_count)/float(335)) )
	h_b = float(0) - ( (float(middle_left_posB_count)/float(205) )*np.log2(float(middle_left_posB_count)/float(205)) + (float(middle_left_negX_count)/float(205))*np.log2(float(middle_left_negX_count)/float(205)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain

def gain_bottom_right():
	h_x = float(0) - ( (float(middle_right_posX_count)/float(418) )*np.log2(float(middle_right_posX_count)/float(418)) + (float(middle_right_negX_count)/float(418))*np.log2(float(middle_right_negX_count)/float(418)) )
	h_o = float(0) - ( (float(middle_right_posO_count)/float(334) )*np.log2(float(middle_right_posO_count)/float(334)) + (float(middle_right_negX_count)/float(334))*np.log2(float(middle_right_negX_count)/float(334)) )
	h_b = float(0) - ( (float(middle_right_posB_count)/float(205) )*np.log2(float(middle_right_posB_count)/float(205)) + (float(middle_right_negX_count)/float(205))*np.log2(float(middle_right_negX_count)/float(205)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain
	
def gain_bottom_middle():
	h_x = float(0) - ( (float(middle_middle_posX_count)/float(378) )*np.log2(float(middle_middle_posX_count)/float(378)) + (float(middle_middle_negX_count)/float(378))*np.log2(float(middle_middle_negX_count)/float(378)) )
	h_o = float(0) - ( (float(middle_middle_posO_count)/float(329) )*np.log2(float(middle_middle_posO_count)/float(329)) + (float(middle_middle_negX_count)/float(329))*np.log2(float(middle_middle_negX_count)/float(329)) )
	h_b = float(0) - ( (float(middle_middle_posB_count)/float(250) )*np.log2(float(middle_middle_posB_count)/float(250)) + (float(middle_middle_negX_count)/float(250))*np.log2(float(middle_middle_negX_count)/float(250)) )
	
	gain = entropy_targetClass() - (float(1)/float(957)) * ( float(417)*(h_x) + float(335)*(h_o) + float(205)*(h_b) )
	
	return gain



gain_array = [gain_top_left(), gain_top_middle(), gain_top_right(), gain_middle_left(), gain_middle_middle(), gain_middle_right(), gain_bottom_left(), gain_bottom_middle(), gain_bottom_right()]
#define X and Y training and testing data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.4, random_state = 100)


#print out training and testing data
'''
	print("X_train :: ", X_train)
	print('')
	print("Y_train :: ", y_train)
	print('')
	print("X_test :: ", X_test)
	print('')
	print("Y_test :: ", y_test)
	print('') 
	'''
	
try:
	print('Probabilty for positive in the target class "outcome" :: ', pos_prob)
	print('')
	print('Probabilty for negative in the target class "outcome" :: ', neg_prob)
	print('')
	print('Entropy for the target class "outcome" :: ', entropy_targetClass())
	print('')
except ValueError, e:
	print("Can't print up the probabilities because :: ", e);	

try:
	print(gain_array)
	print()
except ValueError, e:
	print(e)	

'''

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

'''
