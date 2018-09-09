import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import sys

import xgboost as xgb

if (len(sys.argv) != 2):
	raise Exception('No input files provided')

print('Load data...')
# df_train = pd.read_csv(sys.argv[1], header=None, sep='\t')
# df_test = pd.read_csv(sys.argv[2], header=None, sep='\t')
df = pd.read_csv(sys.argv[1], header=None, sep='\t')
y = df[0].values
X = df.drop(0, axis=1).values

# y_train = df_train[0].values
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train = xgb.DMatrix(X_train, label=y_train)
test = xgb.DMatrix(X_test, label=y_test)
evallist = [(test, 'eval'), (train, 'train')]

print(train)

param = {'max_depth': 5}
num_round = 20

print('Start training...')
bst = xgb.train(param, train, num_round, evallist)


print('Start predicting...')
y_pred = []
y_pred.append(bst.predict(test))



print('The rmse of prediction is:', mean_squared_error(y_test, y_pred[0]) ** 0.5)
