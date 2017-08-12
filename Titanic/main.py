import csv
import tensorflow as tf
from tensorflow.contrib import learn

def ReadDataFile(filename):
	with open(filename, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for i, row in enumerate(spamreader):
			if i == 0:
				res = {j: [] for j, x in enumerate(row)}
				continue
			for j, col in enumerate(row):
				res[j].append(col)
	print('========== {0} loaded.'.format(filename))
	return res

def FormatTrainData(data):
	# male -> 1, female -> 0
	data[4] = [1 if x == 'male' else 0 for x in data[4]]
	# Embark port -> sum of ascii
	data[11] = [sum([ord(x) for x in a]) for a in data[11]]
	# Remove cabin column (10)
	del data[10]
	# Remove ticket column (8)
	del data[8]
	# Remove name column (3)
	del data[3]

	for key in data.keys():
		data[key] = [0 if x == '' else float(x) for x in data[key]]

	for i in range(5):
		print(','.join([str(data[x][i]) for x in data.keys()]))

	# Separate xTrain and yTrain
	yTrain = data[1]
	del data[1]
	print('========== Train data formatted.')
	return [data, yTrain]

def FormatTestData(data):
	# male -> 1, female -> 0
	data[3] = [1 if x == 'male' else 0 for x in data[3]]
	# Embark port -> sum of ascii
	data[10] = [sum([ord(x) for x in a]) for a in data[10]]
	# Remove cabin column (9)
	del data[9]
	# Remove ticket column (7)
	del data[7]
	# Remove name column (2)
	del data[2]

	for key in data.keys():
		data[key] = [0 if x == '' else float(x) for x in data[key]]

	print('========== Test data formatted.')
	return data

def AssembleResult(xTest, yTest):
	print('========== Result assembled.')
	return [[str(int(xTest[i])), str(int(yTest[i]))] for i in range(len(xTest))]

def WriteDataFile(data, filename):
	with open(filename, 'w') as csvfile:
		csvfile.write('PassengerId,Survived')
		for row in data:
			csvfile.write('\n{0},{1}'.format(row[0], row[1]))
	print('========== {0} dumped.'.format(filename))

def Train(xTrain, yTrain, xTest):
	nClasses = 2
	classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("", dimension=len(xTrain.keys()))], n_classes=nClasses)
	classifier.fit(xTrain, yTrain, steps=100)
	yTest = classifier.predict(xTest)
	print('========== Trained.')
	return yTest


if __name__ == '__main__':
	trainData = ReadDataFile('train.csv')
	xTrain, yTrain = FormatTrainData(trainData)

	testData = ReadDataFile('test.csv')
	xTest = FormatTestData(testData)

	yTest = Train(xTrain, yTrain, xTest)
	
	resultData = AssembleResult(xTest, yTest)
	WriteDataFile(resultData)
