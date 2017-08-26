import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


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
    """    
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test).astype(int)
    r2 = regressor.score(X_test, y_test)
    #plot_learning_curve(regressor, "Learning Curves - Linear Regression", X_train, y_train, n_jobs=4, cv = 10)
    #plt.show()
    print("ESSE EH O R²")
    print(r2)
    return regressor, y_pred, r2

train = pd.read_csv('year-prediction-msd-train.txt', header=None)
test = pd.read_csv('year-prediction-msd-test.txt', header=None)

X_train = train.drop([0], axis = 1)
y_train = train.loc[:,0]

X_test = test.drop([0], axis = 1)
y_test = test.loc[:,0]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Linear Regressor 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
regressor = LinearRegression()
LR, y_pred_LR, r2_LR = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Stochastic Gradient Descent Regressor
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import RFE 
regressor = SGDRegressor(learning_rate = 'constant', eta0 = 0.0001, penalty = 'l1', warm_start = True)
SGDReg, y_pred_SGD, r2_SGD = lin_reg(regressor, X_train, y_train, X_test, y_test)

# Fitting SVR to the dataset
#from sklearn.svm import SVR
#regressor = SVR(kernel = 'linear')
#SVReg, y_pred_SVReg, r2_SVReg = lin_reg(regressor, X_train, y_train, X_test, y_test)
'''
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test).astype(int)
r2 = regressor.score(X_test, y_pred)
'''

#from sklearn.metrics import r2_score
#score2 = r2_score(y_test, y_test)

# R squared
#import statsmodels.formula.api as sm
#regressor_OLS = sm.OLS(endog = y_train, exog = X_train).fit()
#regressor_OLS.summary()
