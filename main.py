import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import sys

# Based on the -o flag, run either our version or the original gbt implementation
# Also look for input files, specifically training data and data to predict.
if (len(sys.argv) == 4 and sys.argv[-1] == '-o'):
    from tinygbt_old import Dataset, GBT
    print("Original GBT")
else:
    from tinygbt import Dataset, GBT
    print("GB Projection Tree")
    if (len(sys.argv) != 3):
        raise Exception('No input files provided')

print('Load data...')

# Load train data into df and split the label (first column)
df = pd.read_csv(sys.argv[1], header=None, sep='\t')
y = df[0].values
X = df.drop(0, axis=1).values

# Split into training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Load into dataset objects that GBT uses
train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

# TODO: add params found from hyper parameter tuning
params = {}

gbt = GBT()
gbt.train(params,
          train_data,
          num_boost_round=20,
          valid_set=eval_data,
          early_stopping_rounds=5)

# Load the data to predict
try:
    to_pred = pd.read_csv(sys.argv[2], header=None, sep='\t').values
    y_pred = []

    # Run through our model
    for x in to_pred:
        y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

    # Print the information
    for y in y_pred:
        print(y)

except:
    exit()


# Hyper parameter tuning: tests different configurations of variables such as tree depth
# or min split gain to figure out an optimal configuration.
def hyper_param_tuning():
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

# hyper_param_tuning()