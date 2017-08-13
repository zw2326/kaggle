import csv
import numpy as np
import tensorflow as tf
from pdb import set_trace as bp
from tensorflow.contrib import learn
# Script for Titanic.

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
	return res

def FormatData(data):
	# male -> 1, female -> 0
	data['Sex'] = [1 if x == 'male' else 0 for x in data['Sex']]
	# bp()
	data['Embarked'] = [sum(ord(x) for x in a) for a in data['Embarked']]
	# classMap = lambda x: float(ord('Z') - ord(x))
	# data['Cabin'] = [max(classMap(b[0]) for b in a.split(' ')) if a != '' else classMap('Z') for a in data['Cabin']]
	# print(data['Cabin'])
	# data['Ticket'] = [sum(ord(x) * i for i, x in enumerate(a)) for a in data['Ticket']]
	# bp()
	# del data['Embarked']
	del data['Cabin']
	del data['Ticket']
	del data['Name']

	for key in data.keys():
		data[key] = [0 if x == '' else float(x) for x in data[key]]

	### DEBUG ###
	print(','.join(data.keys()))
	for i in range(5):
		print(','.join([str(data[x][i]) for x in data.keys()]))

	print('========== Data formatted.')
	return data

def FormatTrainData(data):
	data = FormatData(data)

	# Separate xTrain and yTrain
	yTrain = data['Survived']
	del data['Survived']
	return [data, yTrain]

def FormatTestData(data):
	data = FormatData(data)
	return data

def ReadTrainData(filename):
	return FormatTrainData(ReadData(filename))

def ReadTestData(filename):
	return FormatTestData(ReadData(filename))

def WriteData(xTest, yTest, filename):
	res = [[str(int(xTest['PassengerId'][i])), str(int(yTest[i]))] for i in range(len(xTest['PassengerId']))]
	with open(filename, 'w') as csvfile:
		csvfile.write('PassengerId,Survived')
		for row in res:
			csvfile.write('\n{0},{1}'.format(row[0], row[1]))
	print('========== {0} dumped.'.format(filename))

def Train(xTrain, yTrain, xTest):
	nClasses = 2
	xTrainTf = np.array([xTrain[col] for col in sorted(set(xTrain.keys()) - set(['PassengerId']))]).T
	yTrainTf = np.array([yTrain]).T
	xTestTf = np.array([xTest[col] for col in sorted(set(xTest.keys()) - set(['PassengerId']))]).T
	classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=len(xTrainTf))], n_classes=nClasses)
	classifier.fit(xTrainTf, yTrainTf, steps=100)
	yTest = [y for y in classifier.predict(xTestTf)]
	print('========== Trained.')
	return yTest


if __name__ == '__main__':
	xTrain, yTrain = ReadTrainData('train.csv')
	xTest = ReadTestData('test.csv')

	yTest = Train(xTrain, yTrain, xTest)

	WriteData(xTest, yTest, 'output.csv')
