import numpy as np
import pandas as pd

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

### Loading the data sets
columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']
train = pd.read_csv('adult-training.csv', names=columns)
test = pd.read_csv('adult-test.csv', names=columns, skiprows=1)

#print (test.head())
#train.info()


#############Cleaning Data
"""
Cleaning the training and test data by romving invalid data points

Replacing all '?' with NaN and then dropping rows where NaN appears.

"""

train_clean = train.replace(' ?', np.nan).dropna()
test_clean = test.replace(' ?', np.nan).dropna()

#print (test_clean.head())

#print (train_clean["Capital Gain"])

#######STANDARDING
# Fit scaler on train data only. Transform training and testing set
numerical_col = ["Age", "fnlgwt", "Education num", "Capital Gain",
                     "Capital Loss", "Hours/Week"]
scaler = StandardScaler()
train_clean[numerical_col] = scaler.fit_transform(train_clean[numerical_col])
test_clean[numerical_col] = scaler.transform(test_clean[numerical_col])

#print (train_clean["Capital Gain"])


## splitting DataSet
Y_train = train_clean["Income"]
X_train = train_clean.drop("Income", axis=1)

Y_test = test_clean["Income"]
X_test = test_clean.drop("Income", axis=1)
#print (Y_test.head(10))


data = pd.concat([X_train,X_test])
dataEncoded = pd.get_dummies(data)           # One hot Encoding generates over 104 Columns xD

#print (dataEncoded.head())
X_trainEncoded = dataEncoded[:len(X_train)]
X_testEncoded = dataEncoded[len(X_train):]

print (Y_train.head(10))

Y_trainEncoded = Y_train.replace([' <=50K',' >50K' ] , [0,1] )
Y_testEncoded = Y_test.replace([' <=50K.' , ' >50K.'] , [0,1])

#print (Y_trainEncoded.head(10))
#print (Y_testEncoded.head(10))

## Training 
## Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

clf = None
parameters = {}

clf = GaussianNB()


clf_gs = GridSearchCV(clf ,parameters, verbose=1)

clf_gs.fit(X_trainEncoded,Y_trainEncoded)

Y_pred = clf_gs.predict(X_testEncoded)
#print (Y_pred)

print(confusion_matrix(Y_testEncoded,Y_pred))

acc = accuracy_score(Y_testEncoded, Y_pred)

print ("Naive Bayes Approach")
print("Model Accuracy: ",acc*100.0)


lrn = LogisticRegression(penalty = 'l1', C = .001, class_weight='balanced')

lrn.fit(X_trainEncoded, Y_trainEncoded)
Y_pred = lrn.predict(X_testEncoded)
print(confusion_matrix(Y_testEncoded,Y_pred))

acc = accuracy_score(Y_testEncoded, Y_pred)

print ("Logistic Regression Approach")
print("Model Accuracy: ",acc*100.0)

'''

clf = KNeighborsClassifier()
parameters = {"n_neighbors": (3, 5, 6, 8, 10, 15),"weights": ("uniform", "distance")}


clf_gs = GridSearchCV(clf ,parameters, verbose=1)

clf_gs.fit(X_trainEncoded,Y_trainEncoded)

Y_pred = clf_gs.predict(X_testEncoded)
print (Y_pred)

acc = accuracy_score(Y_testEncoded, Y_pred)

print ("K nearest Neighbours Approach")
print("Model Accuracy: ",acc*100.0)

'''

