from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd
import re

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# Load data
oscillators = pd.read_csv("/home/hellsbells/Desktop/oscillator_jacobains.csv")
notOsc = pd.read_csv("/home/hellsbells/Desktop/non-oscillator_jacobians.csv")

osc = []
nonOsc = []

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

# Hacky bullshit to convert strings into list of floats
for i in range(len(oscillators["jacobian"])):
    try:
        j = oscillators["jacobian"][i]
        j = convert_jac_string(j)
        if len(j) != 9:
            continue
        osc.append(j)
    except Exception as e:
        print(e)

for i in range(len(notOsc["jacobian"])):
    try:
        j = notOsc["jacobian"][i]
        j = convert_jac_string(j)
        if len(j) != 9:
            continue
        nonOsc.append(j)
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
#
X_train, X_test, y_train, y_test = train_test_split(j, label, train_size = 0.80, random_state = 100)

# our model instance
model = LogisticRegression()

# train the classifier
model.fit(X_train, y_train)

# use trained model from above to predict the class of "new" data
predicted = model.predict(X_test)

sum = 0
for i in range(len(predicted)):
    if predicted[i] == y_test[i]:
        sum += 1


print(sum)
# # let's see how well the classifier performed
# print(r2_score(y_test, predicted))