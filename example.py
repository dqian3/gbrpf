import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import sys

if (len(sys.argv) == 4 and sys.argv[-1] == '-o'):
    from tinygbt_old import Dataset, GBT
    print("Original GBT")
else:
    from tinygbt import Dataset, GBT
    print("GB Projection Tree")
    if (len(sys.argv) != 3):
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

train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

params = {}

print('Start training...')
for lambd in [0.8]:
    for min_split_gain in [0.1, 0.15]:
        for max_depth in [5, 6]:
            for learning_rate in [0.2, 0.3]:
                params = {'lambda': lambd, 'min_split_gain': min_split_gain, 'max_depth': max_depth, 'learning_rate': learning_rate}
                gbt = GBT()
                gbt.train(params, train_data, num_boost_round=20, valid_set=eval_data, early_stopping_rounds=5)
                y_pred = []
                for x in X_test:
                    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))
                print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5, 'params:', params)
                print()

"""gbt = GBT()    
gbt.train(params,
          train_data,
          num_boost_round=20,
          valid_set=eval_data,
          early_stopping_rounds=5)"""
"""


X_test = pd.read_csv(sys.argv[2], header=None, sep='\t').values
y_pred = []

for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

for y in y_pred:
    print(y)
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
"""