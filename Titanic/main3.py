# End-to-end demonstration of ML pipeline, including generating new features and comparing multiple classification models.
# Refactored to use sklearn, pandas and numpy.
# Using weighted vote. No luck.
# Reference: https://cloud.tencent.com/community/article/229506
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.manifold import Isomap
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import code
import numpy as np
import pandas as pd

def Preprocess(data):
    # Fill NaN. Drop bad column (optional).
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    # data.drop(['PassengerId'], axis=1, inplace=True)

    # Generate feature!
    if 'NameLen' in features:
        data["NameLen"] = data["Name"].apply(len)
    if 'NameClass' in features:
        data["NameClass"] = data["Name"].apply(lambda x: x.split(",")[1]).apply(lambda x :x.split()[0])
        data["NameClass"] = pd.Series(data=pd.factorize(data["NameClass"])[0])
        data["NameClass"] = data["NameClass"].astype(int)
    if 'FamilyNum' in features or 'IsAlone' in features:
        data["FamilyNum"] = data["Parch"] + data["SibSp"] + 1
    if 'Embarked' in features:
        data["Embarked"] = data["Embarked"].map({'S':1,'C':2,'Q':3}).astype(int)
    if 'Sex' in features:
        data["Sex"] = data["Sex"].apply(lambda x: 0 if x == 'male' else 1)
    if 'Fare' in features:
        # data["Fare"] = data["Fare"].apply(lambda x: pd.cut(x, [7.91,14.45,31.0]))
        data["Fare"] = pd.cut(data["Fare"], [0.0,7.91,14.45,31.0,99999.0], labels=[1,2,3,4])
        data["Fare"] = data["Fare"].astype(int)
    if 'Age' in features:
        # data["Age"] = data["Age"].apply(lambda x: pd.cut([18,48,64], x))
        data["Age"] = pd.cut(data["Age"], [0,18,48,64,99999], labels=[1,2,3,4])
    if 'HasCabin' in features:
        data["HasCabin"] = data["Cabin"].apply(lambda x: 0 if pd.isnull(x) else 1)
    if 'IsAlone' in features:
        data["IsAlone"] = data["FamilyNum"].apply(lambda x: 1 if x == 1 else 0)

def Core(train, test):
    # Try out 5(!) models:
    # RandomForestClassifier
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': False,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }
    # ExtraTreesClassifier
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    # AdaBoostClassifier
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }
    # GradientBoostingClassifier
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }
    # SVC
    svc_params = {
        'kernel': 'rbf',
        'C': 0.025
    }

    classifiers = [
        ("Random Forest Classifier", RandomForestClassifier(**rf_params)),
        ("Extra Trees Classifier", ExtraTreesClassifier(**et_params)),
        ("Ada Boost Classifier", AdaBoostClassifier(**ada_params)),
        ("Gradient Boosting Classifier", GradientBoostingClassifier(**gb_params)),
        ("SVC", SVC(**svc_params)),
    ]

    if selfTest == True:
        heldout = [0.95, 0.90, 0.75, 0.50, 0.10]
        rounds = 10 # Number of runs for random model.
        xx = 1. - np.array(heldout)
        yy = {x[0]: [] for x in classifiers} # List of error rate (at each heldout) for each model.

        # Prepare subplot.
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        axMajor = plt.subplot2grid((2, 4), (1, 2), colspan=2)

        for i in range(len(heldout)): # Training progress.
            print 'Heldout: %.2f' % heldout[i]
            xTemp = train[list(features)]
            yTemp = train['Survived']
            xTrain, xTest = train_test_split(xTemp, test_size=heldout[i], random_state=np.random.RandomState(42))
            yTrain, yTest = train_test_split(yTemp, test_size=heldout[i], random_state=np.random.RandomState(42))

            if i == len(heldout) - 1: # Plot actual labels for last heldout.
                xISO = Isomap(n_neighbors=len(features)).fit_transform(xTrain)
                ax[1, 1].scatter(xISO[:, 0], xISO[:, 1], c=yTrain)
                ax[1, 1].set_title('Actual Labels')

            for j, clsTuple in enumerate(classifiers):
                name, clf = clsTuple
                print("    Model: %s" % name)

                # Need avg only for random model.
                yy_ = []
                for r in range(rounds if name == 'Random Forest Classifier' else 1):
                    print '        Round: %d' % (r + 1)
                    clf.fit(xTrain, yTrain)
                    yPred = clf.predict(xTest)
                    yy_.append(1 - np.mean(yPred == yTest)) # Error rate for one run.
                yy[name].append(np.mean(yy_)) # Take avg for all runs as the error rate.

                if i == len(heldout) - 1: # Plot model's labels for last heldout.
                    ax[int(j/4), j%4].scatter(xISO[:, 0], xISO[:, 1], c=clf.predict(xTrain))
                    ax[int(j/4), j%4].set_title(name)

                    # Plot model error rate line for last heldout.
                    axMajor.plot(xx, yy[name], label=name)
                    # Compute model's error matrix for last heldout.
                    ModelScore(xTrain, yTrain, xTest, yTest, name, clf)

        axMajor.legend(loc="upper right")
        axMajor.set_xlabel("Proportion Trained")
        axMajor.set_ylabel("Test Error Rate")
        plt.show()
        return None
    else:
        xTemp = train[list(features)]
        yTemp = train['Survived']
        xTrain, xVoteTrain = train_test_split(xTemp, test_size=0.1, random_state=np.random.RandomState(42))
        yTrain, yVoteTrain = train_test_split(yTemp, test_size=0.1, random_state=np.random.RandomState(42))
        xTest = test[list(features)]

        # Do model training.
        yVotePred = pd.DataFrame()
        yPred = pd.DataFrame()
        for j, clsTuple in enumerate(classifiers):
            name, clf = clsTuple
            print("Model: %s" % name)
            clf.fit(xTrain, yTrain)
            yVotePred[name] = clf.predict(xVoteTrain)
            yPred[name] = clf.predict(xTest)

        # Do vote training.
        weight = [1./len(classifiers) for x in range(len(classifiers))]
        for i in range(len(yVotePred.index)):
            for j in range(len(yVotePred.columns)):
                weight[j] += (1. if yVotePred.iloc[i, j] == yVoteTrain.iloc[i] else -1.) * 0.01
        weight = [weight[i] / sum(weight) for i in range(len(weight))]

        # Do voting.
        
        ret = (yPred.dot(weight) > 0.5).astype(int) # Avoid rounding error.
        return ret

def ModelScore(xTrain, yTrain, xTest, yTest, modelName, model):
    print '\n----------%s----------' % modelName
    if "feature_importances_" in dir(model):
        print 'Feature Importance:'
        print model.feature_importances_
        print ''
    print 'Prediction on training set:'
    print classification_report(yTrain, model.predict(xTrain))
    print 'Prediction on test set:'
    print classification_report(yTest, model.predict(xTest))

def Save(index, yPred):
    fid = open('main3-output.csv', 'w')
    fid.write('PassengerId,Survived')
    for i in range(index.size):
        fid.write('\n{0},{1}'.format(index.iloc[i], yPred.iloc[i]))
    fid.close()


if __name__ == '__main__':
    selfTest = False
    # features = set(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'NameLen', 'NameClass', 'HasCabin', 'FamilyNum', 'IsAlone'])
    features = set(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'NameClass', 'HasCabin', 'FamilyNum', 'IsAlone'])

    # Prepare training and test data.
    train = pd.read_csv('train.csv')
    Preprocess(train)
    test = pd.read_csv('test.csv')
    Preprocess(test)

    # Examine data.
    # print train.describe()
    # print train.describe(include=['O'])
    # print test.describe()
    # print test.describe(include=['O'])

    # Do training and predicting.
    yPred = Core(train, test)

    # Save output.
    if selfTest == False:
        Save(test['PassengerId'], yPred)
