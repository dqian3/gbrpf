import pandas as pd
from sklearn.metrics import mean_squared_error
import re
import sys
import numpy as np

print(sys.argv)

if (len(sys.argv) != 3):
	raise Exception('No trainset file/output provided')

train = pd.read_csv(sys.argv[1], sep=',')
test = pd.read_csv(sys.argv[2], sep=',')
train = train.drop('Id', axis=1)

train_len= len(train)

df = pd.concat(objs=[train, test], axis=0)

df = df.fillna(df.mean())
dropped = df.loc[:,df.columns[-1]]
df = df.drop(df.columns[len(df.columns)-1], axis = 1)
df.insert(0, 0, dropped)
print (df)

df = pd.get_dummies(df)
print (df)

train_output = df[:train_len]
test_output = df[train_len:]

train_output.to_csv('sanitized_train.csv', sep='\t', header=False, index=False)
test_output.to_csv('sanitized_test.csv', sep='\t', header=False, index=False)