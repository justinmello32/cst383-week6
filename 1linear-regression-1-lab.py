#%%
import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

#1
df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/machine.csv")
df.index = df['vendor']+' '+df['model']
df.drop(['vendor', 'model'], axis=1, inplace=True)
df['cs'] = np.round(1e3/df['myct'], 2) # clock speed in MHz (millions of cycles/sec)

#2
#Viewed website, PRP stands for published relative performance, int.

#3
sns.pairplot(df)

#4
sns.scatterplot(data=df,x="mmin",y="cach")

#5
x = df[['prp']].values
y = df['mmin'].values
regr = LinearRegression()
regr.fit(x,y)


# %%
