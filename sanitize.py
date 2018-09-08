import pandas as pd
from sklearn.metrics import mean_squared_error
import re
import sys
import numpy as np

print(sys.argv)

if (len(sys.argv) != 2):
	raise Exception('No trainset file/output provided')

df = pd.read_csv(sys.argv[1], sep=',')

print (df)

df = df.drop('ID', axis=1)

df = df.fillna(df.mean())
dropped = df.loc[:,df.columns[-1]]
df = df.drop(df.columns[len(df.columns)-1], axis = 1)
df.insert(0, 0, dropped)
print (df)

df = pd.get_dummies(df)
print (df)


df.to_csv('sanitized_train.csv', sep='\t', header=False, index=False)
