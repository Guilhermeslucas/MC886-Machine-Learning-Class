import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import RFE



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

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
        y_pred: prediction vector
        r2: R squared 
    """    
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test).astype(int)
    #r2 = regressor.score(X_test, y_test)
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    #plot_learning_curve(regressor, "Learning Curves - Linear Regression", X_train, y_train, n_jobs=4)
    #plt.show()
    return regressor, y_pred, r2, mse

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
train = reduce_dataset(train)

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



# Getting F and P Values
from sklearn.feature_selection import f_regression
F, pval = f_regression(X_train, y_train)

# Excluding features with low P value
X_train = X_train[:, pval > 0.05]
X_test = X_test[:, pval > 0.05]

"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1, random_state = 42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
"""
# Linear Regressor 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
LR, y_pred_LR, r2_LR, mse_LR = lin_reg(selector, X_train, y_train, X_test, y_test)

# Ridge Regression
from sklearn.linear_model import Ridge
regressor = Ridge(alpha = 1000)
selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
Rid, y_pred_Rid, r2_Rid, mse_Rid = lin_reg(regressor, X_train, y_train, X_test, y_test)

# LASSO
from sklearn.linear_model import Lasso
regressor = Lasso(alpha = 0.1, random_state = 42)
selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
Las, y_pred_Las, r2_Las, mse_Las = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Elastic Net
from sklearn.linear_model import ElasticNetCV
regressor = ElasticNetCV(cv=5, random_state = 42)
selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
enet, y_pred_enet, r2_enet, mse_enet = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Stochastic Gradient Descent Regressor
from sklearn.linear_model import SGDRegressor
regressor = SGDRegressor(learning_rate = 'constant', eta0 = 0.0001, penalty = None, warm_start = True)
selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
SGDReg, y_pred_SGD, r2_SGD, mse_SGD = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, n_jobs = 3, random_state = 42, verbose=2, min_samples_leaf = 20, max_features = 0.2)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
RF, y_pred_RF, r2_RF, mse_RF = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.01, criterion ='mse', random_state = 42, verbose = 2)
#selector = RFE(estimator = regressor,  n_features_to_select = 20, step=1, verbose=2)
grad_boost, y_pred_gb, r2_gb, mse_gb = lin_reg(regressor, X_train, y_train, X_test, y_test)
test_score = np.zeros(500, dtype=np.float64)
for i, y_pred in enumerate(grad_boost.staged_predict(X_test)):
    test_score[i] = grad_boost.loss_(y_test, y_pred)

plt.title('Training Error')
plt.plot(np.arange(1,501), grad_boost.train_score_, 'b-', label='Training Set Error')
plt.plot(np.arange(1,501), test_score, 'r-', label='Test Set Error')
plt.legend(loc='upper right')
plt.xlabel('Gradient Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_predict
predic = cross_val_predict(estimator = RF, X = X_train, y = y_train, cv = 5, n_jobs = -1)
predic = predic.astype(int)

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

# Predicting on training set -> creating the model for plot
y_pred_train_LR = LR.predict(X_train).astype(int)
y_pred_train_Rid = Rid.predict(X_train).astype(int)
y_pred_train_enet = enet.predict(X_train).astype(int)
y_pred_train_SGD = SGDReg.predict(X_train).astype(int)
y_pred_train_Las = Las.predict(X_train).astype(int)
y_pred_train_RF = RF.predict(X_train).astype(int)

#Plotting the model
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1, random_state = 42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_pred_train_RF, color='blue')
plt.show()

"""
# R squared
import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
regressor_OLS.summary()
"""
