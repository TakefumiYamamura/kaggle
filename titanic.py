# http://qiita.com/suzumi/items/8ce18bc90c942663d1e6
import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
# print(mpl.matplotlib_fname() + '\matplotlibrc')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv').replace("male",0).replace("female",1)
train["Age"].fillna(train.Age.median(), inplace=True)
# train.describe()
split_data = []
for survived in [0,1]:
    split_data.append(train[train.Survived==survived])
# print split_data

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)
# plt.show()
# x = np.arange(-3, 3, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
temp = [i["Age"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=16)


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train2 = train.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
print train2.dtypes


train_data = train2.values
# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# # Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[:,2:],train_data[:,1])

test_df= pd.read_csv("test.csv").replace("male",0).replace("female",1)

test_df["Age"].fillna(train.Age.median(), inplace=True)
test_df["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test_df2 = test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
# # Take the same decision trees and run it on the test data
# output = forest.predict(test_data)

test_data = test_df2.values
xs_test = test_data[:, 1:]
output = forest.predict(xs_test)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_data[:,0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
