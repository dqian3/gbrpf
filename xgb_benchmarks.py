import xgboost as xgb
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
bos['Price'] = boston.target
X = bos.drop('PRICE', axis=1)
Y = bos['PRICE']
X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
xgbr = xgb.XGBRegressor()
xgbr.fit(X_train, Y_train)
Y_pred = xgbr.predict(X_test)
rmse = sklearn.metrics.root_mean_squared_error(Y_test, Y_pred)
print(rmse)
