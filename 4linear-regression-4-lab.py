import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)# clock speed in MHz

#2
y = df['prp'].values
predictors = ['mmin', 'mmax']
X = df[predictors].values

#3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)

#4
y_predict = reg.predict(X_test)
mse = ((y_predict - y_test)**2).mean()

