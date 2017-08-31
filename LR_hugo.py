import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict

def lin_reg(regressor, X_train, y_train, X_test, y_test):
    """
    Function that takes the regressor, fits and predicts for test set, returning r2 score
    Parameters:
    -----------
        regressor: the linear model created
        X_train: Training set
        y_train: target variable for training set
        X_test: Test set
        y_test: target variable for test set
    Returns:
    --------
        regressor: the fitted regressor
        y_pred_train,: y predict with k fold cross validation for training set
        y_pred_test: y predict for test set
        r2_train: R squared for training set
        mse_train: Mean Squared Error for training set 
        r2_test: R Squared for test set
        mse_test: Mean Squared Error for test set
    """    

    y_pred_train = cross_val_predict(estimator = regressor, X = X_train, y = y_train, cv = 10, n_jobs = 3).astype(int)
    r2_train = r2_score(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    regressor.fit(X_train, y_train)
    #r2_train = regressor.score(X_train, y_train)
    y_pred_test = regressor.predict(X_test).astype(int)
    r2_test = r2_score(y_test, y_pred_test)
    #r2_test = regressor.score(X_test, y_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    metrics = {'r2_train':r2_train,
               'r2_test':r2_train,
               'mse_train':mse_train,
               'mse_test': mse_test}
    preds = {'y_pred_train': y_pred_train,
             'y_pred_test': y_pred_test}
    return regressor, preds, metrics

def reduce_dataset(df):
    """
    Function created in order to make a smaller dataset, trying to avoid overfitting
    Parameters:
    -----------
        df: pandas dataframe you want to filter
    Returns:
    -------
        new_data: smaller dataset
    """
    new_data = pd.DataFrame()
    for year in range(1924,2011):
        rows = train.loc[train[0] == year].head(n=21)
        new_data = new_data.append(rows, ignore_index=True)
    return new_data

train = pd.read_csv('year-prediction-msd-train.txt', header=None)
test = pd.read_csv('year-prediction-msd-test.txt', header=None)

# Sorting by the year
train = train.sort([0], ascending=True)
test = test.sort([0], ascending=True)

#Plotting the histogram of values per year
from collections import Counter
cont = dict(Counter(train[0]))
plt.bar(list(cont.keys()), list(cont.values()))
plt.show()

# Reducing the dataset
#train = reduce_dataset(train)

# Getting the independent variable and the dependent variable
X_train = train.drop([0], axis = 1)
y_train = train.loc[:,0]

X_test = test.drop([0], axis = 1)
y_test = test.loc[:,0]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""
# Getting F and P Values
from sklearn.feature_selection import f_regression
F, pval = f_regression(X_train, y_train)

# Excluding features with low P value
X_train = X_train[:, pval > 0.05]
X_test = X_test[:, pval > 0.05]
"""
"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1, random_state = 42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
"""

# Stochastic Gradient Descent Regressor
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(learning_rate = 'constant', eta0 = 0.00001, penalty = 'l1', warm_start = True, random_state = 42)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
#SGDReg, y_pred_train_SGD, y_pred_test_SGD, r2_train_SGD, mse_train_SGD, r2_test_SGD, mse_test_SGD = lin_reg(regressor, X_train, y_train, X_test, y_test)
SGDReg, preds_SGD, metrics_SGD = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Linear Regressor 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
LR, preds_LR, metrics_LR = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Ridge Regression
from sklearn.linear_model import Ridge
regressor = Ridge(alpha = 1000, solver = 'sparse_cg', random_state = 42)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
Rid, preds_Rid, metrics_Rid = lin_reg(regressor, X_train, y_train, X_test, y_test)

# LASSO
from sklearn.linear_model import Lasso
regressor = Lasso(alpha = 0.1, random_state = 42)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
Las, preds_Las, metrics_Las = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.01, criterion ='mse', random_state = 42, verbose = 2)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
GB, preds_GB, metrics_GB = lin_reg(regressor, X_train, y_train, X_test, y_test)
test_score = np.zeros(500, dtype=np.float64)
for i, y_pred in enumerate(GB.staged_predict(X_test)):
    test_score[i] = GB.loss_(y_test, y_pred)

plt.title('Training Error')
plt.plot(np.arange(1,501), GB.train_score_, 'b-', label='Training Set Error')
plt.plot(np.arange(1,501), test_score, 'r-', label='Test Set Error')
plt.legend(loc='upper right')
plt.xlabel('Gradient Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, n_jobs = 3, random_state = 42, verbose=2, min_samples_leaf = 20, max_features = 0.2)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
RF, preds_RF, metrics_RF = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[150], 'min_samples_leaf': [5, 15, 30, 50], 'max_features': [0.1, 0.15, 0.2]}]
grid_search_RF = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs = 3,
                           verbose=2)
grid_search_RF = grid_search_RF.fit(X_train, y_train)
best_accuracy = grid_search_RF.best_score_
best_parameters = grid_search_RF.best_params_

#Plotting the model
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1, random_state = 42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_pred_train_RF, color='blue')
plt.show()

