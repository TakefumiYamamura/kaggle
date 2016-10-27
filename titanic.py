import pandas as pd
import matplotlib as mpl
mpl.use('tkagg')
print(mpl.get_configdir() + '/matplotlibrc')
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv('train.csv').replace("male",0).replace("female",1)
train["Age"].fillna(train.Age.median(), inplace=True)
# train.describe()
split_data = []
for survived in [0,1]:
    split_data.append(train[train.Survived==survived])

temp = [i["Pclass"].dropna() for i in split_data]
plt.hist(temp, histtype="barstacked", bins=3)

x = np.arange(-3, 3, 0.1)
y = np.sin(x)
plt.plot(x, y)
