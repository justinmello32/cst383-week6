import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2)# clock speed in MHz 

#2
y = np.array(df['prp'])
x = np.array(df['mmin'],df['mmax'])

#3
x_train, x_test, y_train, y_test = train_test_split(x,y)

#4
#I used the mmin and mmax values for y

#5
#I believe so, I tried experminiting values of df to compare results.