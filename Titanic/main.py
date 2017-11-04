import csv
import numpy as np
import sys
import tensorflow as tf
from pdb import set_trace as bp
from tensorflow.contrib import learn
# Script for Titanic.

isSelfTest = True
nSelfTest = 100

def FormatData(data):
	# male -> 1, female -> 0
	data['Sex'] = [1 if x == 'male' else 0 for x in data['Sex']]

	# bp()
	### data['Embarked'] = [sum(ord(x) for x in a) for a in data['Embarked']]

	### classMap = lambda x: float(ord('Z') - ord(x))
	### data['Cabin'] = [max(classMap(b[0]) for b in a.split(' ')) if a != '' else classMap('Z') for a in data['Cabin']]

	### data['Ticket'] = [sum(ord(x) * i for i, x in enumerate(a)) for a in data['Ticket']]
	# bp()

	# del data['Pclass']
	del data['Name']
	del data['Age'] # many rows do not have it and we cannot guess
	del data['Ticket']
	del data['Fare']
	del data['Cabin']
	del data['Embarked']

	for key in data.keys():
		if key in ['Cabin']: # allow null data for these columns
			data[key] = [0 if x == '' else float(x) for x in data[key]]
		else: # do not allow null data
			try:
				data[key] = [float(x) for x in data[key]]
			except Exception as e:
				print('key = {0}'.format(key))
				print(e)
				exit(1)

	### DEBUG ###
	print(','.join(data.keys()))
	for i in range(5):
		print(','.join([str(data[x][i]) for x in data.keys()]))
	### DEBUG ###

	print('========== Data formatted.')
	return data

def ReadData(filename):
	with open(filename, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i, row in enumerate(spamreader):
			if i == 0:
				res = {x: [] for x in row}
				ind2col = {j: x for j, x in enumerate(row)} # map column index to column name
				continue
			for j, col in enumerate(row):
				res[ind2col[j]].append(col)
	print('========== Data read.'.format(filename))
	FormatData(res)
	return res

def SeparateData(xyTrain):
	xyTest = {k: xyTrain[k][:nSelfTest] for k in xyTrain.keys()}
	xyTrain = {k: xyTrain[k][nSelfTest:] for k in xyTrain.keys()}
	return [xyTrain, xyTest]

def Train(xyTrain, test):
	nClasses = 2
	xTrainTf = np.array([xyTrain[col] for col in sorted(set(xyTrain.keys()) - set(['PassengerId', 'Survived']))]).T
	yTrainTf = np.array(xyTrain['Survived']).T
	xTestTf = np.array([test[col] for col in sorted(set(test.keys()) - set(['PassengerId', 'Survived']))]).T
	classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=len(xTrainTf))], n_classes=nClasses)
	classifier.fit(xTrainTf, yTrainTf, steps=100)
	yTest = [y for y in classifier.predict(xTestTf)]
	print('========== Trained.')
	return yTest

# For self test.
def Compare(yTestResult, xyTest):
	nGood = sum(xyTest['Survived'][i] == yTestResult[i] for i in range(len(yTestResult)))
	print('========== Compared.')
	print('Accuracy: {0}'.format(float(nGood) / nSelfTest))

# For real test.
def WriteData(xTest, yTest, filename):
	res = [[str(int(xTest['PassengerId'][i])), str(int(yTest[i]))] for i in range(len(xTest['PassengerId']))]
	with open(filename, 'w') as csvfile:
		csvfile.write('PassengerId,Survived')
		for row in res:
			csvfile.write('\n{0},{1}'.format(row[0], row[1]))
	print('========== {0} dumped.'.format(filename))


if __name__ == '__main__':
	for i in sys.argv[1:]:
		exec(i)

	if isSelfTest:
		xyTrain = ReadData('train.csv')
		xyTrain, xyTest = SeparateData(xyTrain)
		yTestResult = Train(xyTrain, xyTest)
		Compare(yTestResult, xyTest)
	else:
		xyTrain = ReadData('train.csv')
		xTest = ReadData('test.csv')
		yTestResult = Train(xyTrain, xTest)
		WriteData(xTest, yTestResult, 'main-output.csv')
