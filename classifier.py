from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from random import sample

import pandas as pd
import numpy as np

# Load data
oscillators = pd.read_csv("/home/hellsbells/Desktop/ID-2oscillator_jacobians2.csv")
notOsc = pd.read_csv("/home/hellsbells/Desktop/ID-2non-oscillator_jacobians2.csv")

osc = []
nonOsc = []
ID = []

def convert_jac_string(j):
    j = re.sub('\s{1,}', '$', j)
    j = j[1:-1]
    if j.startswith("$"):
        j = j[1:]
    if j.endswith("$"):
        j = j[:-1]
    j = j.split("$")

    newJ = []
    for num in j:
        if 'e' in num:
            if '+' in num:
                num, e = num.split('e')
                e = e[1:]
                newJ.append(float(num) * 10 ** (float(e)))
            elif '-' in num:
                num, e = num.split('e')
                e = e[1:]
                newJ.append(float(num) * 10 ** (-1 * float(e)))
        else:
            newJ.append(float(num))
    return newJ

def custom_train_test_split(j, label, ID, train_portion=0.80):
    train_size = int(np.round(train_portion*len(j)))
    X_train, y_train, ID_train = [], [], []
    X_test, y_test,  ID_test = [], [], []
    train_indices = sample(range(len(label)), train_size)
    test_indices = [x for x in range(len(label)) if x not in train_indices]
    for i in train_indices:
        X_train.append(j[i])
        y_train.append(label[i])
        ID_train.append(ID[i])
    for i in test_indices:
        X_test.append(j[i])
        y_test.append(label[i])
        ID_test.append(ID[i])
    return X_train, X_test, y_train, y_test, ID_train, ID_test




# Hacky bullshit to convert strings into list of floats
for i in range(len(oscillators["jacobian"])):
    try:
        j = oscillators["jacobian"][i]
        j = convert_jac_string(j)
        if len(j) != 9:
            continue
        osc.append(j)
        ID.append(oscillators["ID"][i][1:])
    except Exception as e:
        print(e)

for i in range(len(notOsc["jacobian"])):
    try:
        j = notOsc["jacobian"][i]
        j = convert_jac_string(j)
        if len(j) != 9:
            continue
        nonOsc.append(j)
        ID.append(notOsc["ID"][i][1:])
    except Exception as e:
        print(e)

j = []
label = []

for i in range(len(osc)):
    j.append(osc[i])
    label.append(True)

for i in range(len(nonOsc)):
    j.append(nonOsc[i])
    label.append(False)


accuracy = []
from sklearn import svm
for i in range(1):

    # X_train, X_test, y_train, y_test = train_test_split(j, label, train_size = 0.80)#, random_state = 100)
    X_train, X_test, y_train, y_test, ID_train, ID_test = custom_train_test_split(j, label, ID)


    ## Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    # # SVM
    # clf = svm.SVC()
    # clf.fit(X_train, y_train)
    # predicted = clf.predict(X_test)

    # # Logistic Regression
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # predicted = model.predict(X_test)

    # ## Naive Bayes
    # gnb = GaussianNB()
    # predicted = gnb.fit(X_train, y_train).predict(X_test)
    #

    mispredicted = []
    sum = 0
    for i in range(len(predicted)):
        if predicted[i] == y_test[i]:
            sum += 1
        else:
            mispredicted.append(ID_test[i])
    accuracy.append(sum/len(predicted))


print(np.mean(accuracy))
print(np.std(accuracy))

for id in mispredicted:
    result = mm.query_database({"ID": id})
    ant = result[0]["model"]
    m = te.loada(ant)
    r = m.simulate()
    m.plot()
    input("press any key to continue")
