import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pandas.set_option('display.max_rows', 100)
pandas.set_option('display.max_columns', 100)
sns.set()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_Id = test_df['Id']
super_df = train_df.append(test_df)
combine_df = [train_df, test_df, super_df]


#
# print(super_df.info())
# print(super_df.isnull().sum().sort_values(ascending=False))
# print(super_df.nunique())

def boxcoxtransform(*column_names, add=0):
    for colname in column_names:
        bc_xform_values, bc_lambda = stats.boxcox(super_df[colname] + add)
        #print('BoxCox Transform ', colname, ' with lambda: ', bc_lambda)
        for df in combine_df:
            df[colname] = stats.boxcox(df[colname] + add, bc_lambda)

def ZScoreScale(*column_names, clip=1000):
    for colname in column_names:
        mean = super_df[colname].dropna().mean()
        std = super_df[colname].dropna().std()
        for df in combine_df:
            df[colname] = (df[colname] - mean) / std
            df[colname] = df[colname].clip(lower=-clip, upper=clip)


def numericcolinfo(colname, plots=True):
    print('\nColumn info for: ', colname)
    print(super_df[colname].describe())
    print('NUnique: ', super_df[colname].nunique())
    print('Isnull: ', super_df[colname].isnull().sum())
    print('IsZero: ', (super_df[colname] == 0).sum())
    if plots == True:
        #ax1 = sns.boxplot(y=colname, data=super_df)
        ax2 = sns.distplot(super_df[colname],rug=True)
        ax3 = sns.jointplot(colname, 'SalePrice', data=train_df)
        plt.show()


def categorycolinfo(colname, plots=True):
    print('\nColumn info for: ', colname)
    n_unique = super_df[colname].nunique()
    print('NUnique: ', n_unique)
    print('Isnull: ', super_df[colname].isnull().sum())
    print(super_df[colname].value_counts())


# Sales Price
# Some extremely high values in SalesPrice. Let's clip
# print(train_df['SalePrice'].describe(percentiles=[0.995]))
SP_clip_upper = 500000
for df in [train_df, super_df]:
    df['SalePrice'] = df['SalePrice'].clip(upper=SP_clip_upper)
    df['SalePriceln'] = np.log1p(df['SalePrice'])

numeric_feature_columns = []
category_feature_columns = []

# OverallQual
# numericcolinfo('OverallQual')
ZScoreScale('OverallQual')
numeric_feature_columns.append('OverallQual')

# GrLivArea
#for df in combine_df:
#    df['GrLivArea'] = df['GrLivArea'].clip(lower=-3, upper=3)
boxcoxtransform('GrLivArea')
ZScoreScale('GrLivArea')
numeric_feature_columns.append('GrLivArea')

# All the Garage variables

# GarageType
# categorycolinfo('GarageType')

# Having a garage or not, Garage Area, and Garage # of cars seem to be the most usable

for df in combine_df:
    df['noGarage'] = df['GarageArea'] == 0
numeric_feature_columns.append('noGarage')
# numericcolinfo('noGarage')

# GarageArea

# Found 1 NA in GarageArea in test_df. Let's fill it based on the Garage Type median
# print(test_df.loc[test_df['GarageArea'].isnull(),'GarageType'])
# Found it to be Detached garage type. Use that to fill the NA
fill_GarageArea = super_df.loc[super_df['GarageType'] == 'Detchd', 'GarageArea'].dropna().median()

for df in combine_df:
    df.loc[df['GarageArea'].isnull(), 'GarageArea'] = fill_GarageArea

boxcoxtransform('GarageArea', add=1)
ZScoreScale('GarageArea', clip=3)
# numericcolinfo('GarageArea')
numeric_feature_columns.append('GarageArea')

# GarageCars
# Clip to 4 Garages (before scaling) and fill one NA with 0
for df in combine_df:
    df['GarageCars'] = df['GarageCars'].clip(lower=0, upper=3).fillna(value=0)
ZScoreScale('GarageCars')
# numericcolinfo('GarageCars')
numeric_feature_columns.append('GarageCars')

# TotalBsmtSF
# Obviously a high correlation. Will also look at all the other Basement fields here
# basement_cat_fields = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
# basement_num_fields = ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
# for col in basement_cat_fields:
#    categorycolinfo(col)
# for col in basement_num_fields:
#    numericcolinfo(col, False)

# Found 1 null in Test. ALl other Basement fields were Nan for it, so filled TotalSF with 0
# print('Null BsmtSF: ',test_df['TotalBsmtSF'].isnull().sum())
# print(test_df.loc[test_df['TotalBsmtSF'].isnull(), basement_cat_fields])

for df in combine_df:
    df['TotalBsmtSF'].fillna(value=0, inplace=True)
    df['hasBasement'] = df['TotalBsmtSF'] == 0

numeric_feature_columns.append('hasBasement')

boxcoxtransform('TotalBsmtSF', add=1)
ZScoreScale('TotalBsmtSF', clip=3)
# numericcolinfo('TotalBsmtSF')
numeric_feature_columns.append('TotalBsmtSF')

# 1stFlr & 2ndFlr SF
# numericcolinfo('1stFlrSF')
boxcoxtransform('1stFlrSF', add=1)
ZScoreScale('1stFlrSF', clip=3)
# numericcolinfo('1stFlrSF')
numeric_feature_columns.append('1stFlrSF')


# Bathrooms
for df in combine_df:
    df['TotalBaths'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']
    df['TotalBaths'] = df['TotalBaths'].fillna(value=0).clip(upper=4)

ZScoreScale('TotalBaths')
numeric_feature_columns.append('TotalBaths')

# Rooms
# numericcolinfo('TotRmsAbvGrd')

for df in combine_df:
    df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].clip(lower=3, upper=10)

ZScoreScale('TotRmsAbvGrd')
numeric_feature_columns.append('TotRmsAbvGrd')

# Yr Built / Remod

# sns.jointplot('YearRemodAdd', 'YearBuilt', data=super_df)
# plt.show()
#boxcoxtransform('YearBuilt')
ZScoreScale('YearBuilt')
numeric_feature_columns.append('YearBuilt')
#boxcoxtransform('YearRemodAdd')
ZScoreScale('YearRemodAdd')
numeric_feature_columns.append('YearRemodAdd')

# Sale info & other info

replace_map = {'Functional': {'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 8}}

for df in combine_df:
    df['SaleNew'] = df['SaleType'] == 'New'
    df['SaleAbnormal'] = df['SaleCondition'] == 'Abnorml'
    df.replace(replace_map, inplace=True)

numeric_feature_columns.append('SaleNew')
numeric_feature_columns.append('SaleAbnormal')

# corrmat = train_df.corr().abs()['SalePrice'].sort_values()
# print(corrmat)

# print(numeric_feature_columns)
# print(category_feature_columns)


#for col in numeric_feature_columns:
#   plt.clf()
#   sns.distplot(train_df[col])
#   plt.show()


# Let's start with a simple Regression Model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score






X = train_df[numeric_feature_columns]
y = train_df['SalePriceln']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


ridge_params = {
    'alpha': np.arange(0.25,2,0.25),
}
lasso_params = {
    'alpha': np.arange(0.25,2,0.25),
}
elasticnet_params = {
    'alpha': np.arange(0.25,2,0.25),
    'l1_ratio': np.arange(0.0,1,0.25)
}
sgdregressor_params = {
    'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.007, 0.01, 0.03],
    'penalty': ['l1', 'l2', 'elasticnet']
}
svr_params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': np.arange(0.25, 2, 0.25)
}
knn_params = {
    'n_neighbors': np.arange(3,10,2),
    'weights': ['uniform', 'distance'],
    'p': np.arange(1,2,0.25)
}
dt_params = {
    'criterion': ['mse', 'friedman_mse', 'mae'],
    'max_depth': np.arange(1,50,5)
}


models_list = [
    ('LR', LinearRegression(), {}),
    ('Ridge', Ridge(), ridge_params),
    ('Lasso', Lasso(), lasso_params),
    ('ElasticNet', ElasticNet(), elasticnet_params),
    ('SGDRegressor', SGDRegressor(), sgdregressor_params),
    ('SVR', SVR(), svr_params),
    ('KNN', KNeighborsRegressor(), knn_params),
    ('GaussianProcess', GaussianProcessRegressor(), {}),
    ('DTree', DecisionTreeRegressor(), dt_params)
]



rmsle_scores = []
r2_scores = []
model_names = []
best_estimators = []

for name, model, model_params in list(models_list):
    print('-'*100)
    print('Fitting ', name)
    model_names.append(name)
    model_grid = GridSearchCV(estimator=model, param_grid=model_params, scoring='neg_root_mean_squared_error',
                            verbose=0, n_jobs=-1, cv=5)
    model_grid.fit(X_train, y_train)
    print(name, ' Params: ', model_grid.best_params_)
    print(name, ' Score: ', -1* model_grid.best_score_)

    y_pred = model_grid.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test,y_pred))
    r2score = r2_score(y_test, y_pred)
    print(name, ' Test RMSLE: ', score)
    print(name, ' Test R2: ', r2score)
    rmsle_scores.append(score)
    r2_scores.append(r2score)
    best_estimators.append(model_grid.best_estimator_)

model_summary = pd.DataFrame(data=rmsle_scores, index=model_names, columns=['RMSLE'])
model_summary['R2'] = r2_scores
model_summary['Models'] = best_estimators
print(model_summary[['RMSLE','R2']].sort_values(by='RMSLE'))

final_model = model_summary.loc['Ridge','Models']

y_pred_final = final_model.predict(test_df[numeric_feature_columns])
y_pred_final_exp = np.expm1(y_pred_final)

submission = pd.DataFrame({'Id': test_Id, 'SalePrice': y_pred_final_exp})
filename = 'Submission 3 - Tuned Ridge Regression.csv'
submission.to_csv(filename, index=False)
print('\nSubmission file created: ', filename)

















