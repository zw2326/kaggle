# ML full pipeline.
# Reference:
# https://cloud.tencent.com/community/article/229506
# http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# http://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb
# https://www.kaggle.com/shivendra91/rolling-in-the-deep
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

from sklearn import preprocessing

from sklearn.manifold import Isomap
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import code
import numpy as np
import pandas as pd

def Examine(data):
    # Correlate survival with age.
    '''
    figure = plt.figure(figsize=(15, 8))
    # data['Age'].fillna(data['Age'].median(), inplace=True)
    plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
             bins = 30,label = ['Survived','Dead'])
    plt.xlabel('Age')
    plt.ylabel('Number of passengers')
    plt.legend()
    '''

    # Correlate survival with sex.
    '''
    survived_sex = data[data['Survived']==1]['Sex'].value_counts()
    dead_sex = data[data['Survived']==0]['Sex'].value_counts()
    df = pd.DataFrame([survived_sex,dead_sex])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(15,8))
    '''

    # Correlate survival with fare.
    figure = plt.figure(figsize=(15, 8))
    plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color=['g','r'], bins=30, label=['Survived','Dead'])
    plt.xlabel('Fare')
    plt.ylabel('Number of passengers')
    plt.legend()

    plt.show()
    exit(0)

def ProcessTitle(data):
    # Extract the title from each name.
    data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    data.drop('Name', axis=1, inplace=True)

    # A map of more aggregated titles.
    titleDict = {
        "Capt":         "Officer",
        "Col":          "Officer",
        "Major":        "Officer",
        "Jonkheer":     "Royalty",
        "Don":          "Royalty",
        "Sir" :         "Royalty",
        "Dr":           "Officer",
        "Rev":          "Officer",
        "the Countess": "Royalty",
        "Dona":         "Royalty",
        "Mme":          "Mrs",
        "Mlle":         "Miss",
        "Ms":           "Mrs",
        "Mr" :          "Mr",
        "Mrs" :         "Mrs",
        "Miss" :        "Miss",
        "Master" :      "Master",
        "Lady" :        "Royalty"
    }
    # Map title to aggregated title.
    data['Title'] = data['Title'].map(titleDict)

    # Dummy encoding.
    titlesDummies = pd.get_dummies(data['Title'], prefix='Title')
    data = pd.concat([data, titlesDummies], axis=1)
    return data

# A function that fills the missing values of the Age variable.
def ProcessAge(data):
    rf_params = {
        'n_estimators': 10,
        'max_features': 'sqrt',
        'max_depth': 6,
        'min_samples_leaf': 2,
        'min_samples_split': 10,
        'bootstrap': True
    }
    name, clf = "Random Forest Classifier", RandomForestClassifier(**rf_params)
    rounds = 31

    data = GroupAge(data) # Generate feature 'AgeGroup'
    features = list(set(data.columns) - set(['Survived', 'Age', 'Title', 'Embarked', 'Cabin', 'Pclass', 'Ticket', 'AgeGroup', 'Sex']))
    train = data[pd.notnull(data['AgeGroup'])]
    xTrain = train[features]
    yTrain = train['AgeGroup'].astype('|S6')

    print('Fitting Age')
    print("    Model: %s" % name)
    clfs = []
    for r in range(rounds):
        print '    Round: %d' % (r + 1)
        clf.fit(xTrain, yTrain)
        clfs.append(clf)
    # For NaN value in Age, let all the trained classifiers do voting. Use mode of the voting as the age.
    data['Age'] = data.apply(lambda r: float(pd.Series([clf.predict(r[features].to_frame().T)[0] for clf in clfs]).mode()[0]) if np.isnan(r['Age']) else r['Age'], axis=1)
    return data

def GroupAge(data):
    grouped_train = data.groupby(['Sex', 'Pclass', 'Title'])
    grouped_median_train = grouped_train.median()

    def MapAgeToGroupMedian(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    data['AgeGroup'] = data.apply(lambda r: MapAgeToGroupMedian(r, grouped_median_train) if not np.isnan(r['Age']) else r['Age'], axis=1)
    return data

def ProcessEmbarked(data):
    # Fill with mode, since NaN is very rare.
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Dummy encoding.
    embarkedDummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = pd.concat([data, embarkedDummies], axis=1)
    return data

def ProcessCabin(data):
    # Replace missing cabins with U (for unknown).
    data['Cabin'].fillna('U', inplace=True)

    # Map each Cabin value to the cabin initial.
    data['Cabin'] = data['Cabin'].map(lambda x: x[0])

    # Dummy encoding.
    cabinDummies = pd.get_dummies(data['Cabin'], prefix='Cabin')
    data = pd.concat([data, cabinDummies], axis=1)
    return data

def ProcessSex(data):
    # Map string to numeric.
    data['SexTemp'] = data['Sex'].map({'male': 1,'female': 0})
    return data

def ProcessFare(data):
    # Replace missing fares with mean.
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    return data

def ProcessPclass(data):
    # Dummy encoding.
    pclassDummies = pd.get_dummies(data['Pclass'], prefix="Pclass")
    data = pd.concat([data, pclassDummies], axis=1)
    return data

def ProcessTicket(data):
    # Return ticket prefix or XXX (if prefix not found).
    def CleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'

    # Dummy encoding.
    data['Ticket'] = data['Ticket'].map(CleanTicket)
    ticketsDummies = pd.get_dummies(data['Ticket'], prefix='Ticket')
    data = pd.concat([data, ticketsDummies], axis=1)
    return data

def ProcessFamily(data):
    # Generate family size feature.
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

    # Generate other features based on the family size.
    data['Singleton'] = data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    data['SmallFamily'] = data['FamilySize'].map(lambda s: 1 if 2 <= s and s <= 4 else 0)
    data['LargeFamily'] = data['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return data

def ProcessData(data):
    data = ProcessTitle(data)
    data = ProcessEmbarked(data)
    data = ProcessCabin(data)
    data = ProcessSex(data)
    data = ProcessFare(data)
    data = ProcessPclass(data)
    data = ProcessTicket(data)
    data = ProcessFamily(data)
    data = ProcessAge(data)
    data['Sex'] = data['SexTemp']
    data.drop(['SexTemp', 'Title', 'Embarked', 'Cabin', 'Pclass', 'Ticket', 'AgeGroup'], axis=1, inplace=True)
    # Normalize data.
    #dataTmp = pd.DataFrame(preprocessing.normalize(data, axis=0, copy=True), columns=data.columns)
    #dataTmp['PassengerId'] = data['PassengerId']
    #data = dataTmp
    return data

def Core(train, test):
    # Try out 5(!) models:
    # RandomForestClassifier
    '''
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
    '''
    rf_params = {
        'n_estimators': 10,
        'max_features': 'sqrt',
        'max_depth': 6,
        'min_samples_leaf': 2,
        'min_samples_split': 10,
        'bootstrap': True
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
        #("Extra Trees Classifier", ExtraTreesClassifier(**et_params)),
        #("Ada Boost Classifier", AdaBoostClassifier(**ada_params)),
        #("Gradient Boosting Classifier", GradientBoostingClassifier(**gb_params)),
        #("SVC", SVC(**svc_params)),
    ]

    if gridSearch == True: # Do grid search for Random Forest Classifier parameters.
        heldout = 0.1
        xTemp = train[list(features)]
        yTemp = train['Survived']
        xTrain, xTest = train_test_split(xTemp, test_size=heldout, random_state=np.random.RandomState(42))
        yTrain, yTest = train_test_split(yTemp, test_size=heldout, random_state=np.random.RandomState(42))

        parameter_grid = {
            'max_depth' : [4, 6, 8],
            'n_estimators': [50, 10],
            'max_features': ['sqrt', 'auto', 'log2'],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 3, 10],
            'bootstrap': [True, False],
        }
        forest = RandomForestClassifier()
        cross_validation = StratifiedKFold(yTrain, n_folds=5)

        grid_search = GridSearchCV(forest,
                                   scoring='accuracy',
                                   param_grid=parameter_grid,
                                   cv=cross_validation)

        grid_search.fit(xTrain, yTrain)

        print('Best score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))
        exit(1)

    if hack == True: # Use the best classifier when doing self test to do the actual prediction.
        heldout = 0.1
        rounds = 3999 # Number of runs for random model.

        xTemp = train[list(features)]
        yTemp = train['Survived']
        xTrain, xTest = train_test_split(xTemp, test_size=heldout, random_state=np.random.RandomState(42))
        yTrain, yTest = train_test_split(yTemp, test_size=heldout, random_state=np.random.RandomState(42))

        name, clf = classifiers[0]
        print("Model: %s" % name)

        bestClf = None
        bestErrorRate = 1
        for r in range(rounds if name == 'Random Forest Classifier' else 1):
            print '    Round: %d' % (r + 1)
            clf.fit(xTrain, yTrain)
            yPred = clf.predict(xTest)
            errorRate = 1 - np.mean(yPred == yTest)
            if errorRate < bestErrorRate: # Error rate for one run.
                bestClf = [clf]
                bestErrorRate = errorRate
            elif errorRate == bestErrorRate:
                bestClf.append(clf)
            print 'Best error rate: {0}'.format(bestErrorRate)
            print 'Best clf #: {0}'.format(len(bestClf))

        xTrain = train[list(features)]
        xTest = test[list(features)]
        yTrain = train['Survived']

        print 'Using best classifier'
        yPred = pd.DataFrame()
        for i, clf in enumerate(bestClf):
            clf.fit(xTrain, yTrain)
            yPred[i] = clf.predict(xTest)
        # Do voting (only effective for random model).
        ret = yPred.mode(axis=1)
        return ret[0]

    if selfTest == True:
        heldout = [0.95, 0.90, 0.75, 0.50, 0.10]
        rounds = 299 # Number of runs for random model.
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
        rounds = 299 # Number of runs for random model.

        xTrain = train[list(features)]
        xTest = test[list(features)]
        yTrain = train['Survived']

        name, clf = classifiers[0]
        print("Model: %s" % name)

        yPred = pd.DataFrame()
        for r in range(rounds if name == 'Random Forest Classifier' else 1):
            print '    Round: %d' % (r + 1)
            clf.fit(xTrain, yTrain)
            yPred[r] = clf.predict(xTest)
        # Do voting (only effective for random model).
        ret = yPred.mode(axis=1)

        return ret[0]

def ModelScore(xTrain, yTrain, xTest, yTest, modelName, model):
    print '\n----------%s----------' % modelName
    if "feature_importances_" in dir(model):
        print 'Feature Importance:'
        print model.feature_importances_
        print ''
        features = pd.DataFrame()
        features['feature'] = xTrain.columns
        features['importance'] = model.feature_importances_
        features.sort_values(by=['importance'], ascending=False, inplace=True)
        features.set_index('feature', inplace=True)
        features.plot(kind='barh', figsize=(8, 4))
    print 'Prediction on training set:'
    print classification_report(yTrain, model.predict(xTrain))
    print 'Prediction on test set:'
    print classification_report(yTest, model.predict(xTest))

def Save(index, yPred):
    fid = open('main4-output.csv', 'w')
    fid.write('PassengerId,Survived')
    for i in range(index.size):
        fid.write('\n{0},{1}'.format(index.iloc[i], yPred[i]))
    fid.close()


if __name__ == '__main__':
    examineData = False
    selfTest = False
    gridSearch = False
    hack = False

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    if examineData == True:
        Examine(train)

    train = ProcessData(train)
    test = ProcessData(test)
    missingFeatures = list(set(train.columns) - set(test.columns) - set(['Survived']))
    test = pd.concat([test, pd.DataFrame(columns=missingFeatures)], axis=1)
    test[missingFeatures] = test[missingFeatures].fillna(0)

    #features = list(set(train.columns) - set(['Survived']))
    features = ['PassengerId', 'SibSp', 'LargeFamily', 'Pclass_1', 'SmallFamily', 'FamilySize', 'Cabin_U', 'Age', 'Title_Miss', 'Pclass_3', 'Title_Mrs', 'Fare', 'Sex', 'Title_Mr']

    yPred = Core(train, test)

    if selfTest == False:
        Save(test['PassengerId'], yPred)
