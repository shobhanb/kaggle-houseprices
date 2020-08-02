
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis, boxcox
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


from sklearn.base import BaseEstimator
class RegressorSwitcher(BaseEstimator):

    def __init__(
        self,
        estimator=LinearRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)




pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 15)
pd.options.display.float_format = '{:20,.6f}'.format
sns.set()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_Id = test_df['Id']
super_df = train_df.append(test_df)
combine_df = [train_df, test_df]


def numericcolinfo(colname, plots=True):
    print('\nColumn info for: ', colname)
    print(super_df[colname].describe())
    print('NUnique: ', super_df[colname].nunique())
    print('Isnull: ', super_df[colname].isnull().sum())
    print('IsZero: ', (super_df[colname] == 0).sum())
    print('Skew: ', skew(super_df[colname]))
    print('Kurtosis: ', kurtosis(super_df[colname]))
    if plots == True:
        ax2 = sns.distplot(super_df[colname],rug=True, fit=norm)
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
SP_clip_upper = 600000
for df in [train_df, super_df]:
    df['SalePrice'] = df['SalePrice'].clip(upper=SP_clip_upper)
    df['SalePriceln'] = np.log1p(df['SalePrice'])

#numericcolinfo('SalePrice')
#numericcolinfo('SalePriceln')

for df in combine_df:
    df['TotalBaths'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']
    df['TotalBaths'] = df['TotalBaths'].fillna(value=0).clip(upper=4)


numeric_power_transform_columns = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'LotArea']
category_feature_columns = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley','LotShape', 'LandContour', 'Utilities', 'LotConfig',
    'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'OverallQual', 'OverallCond', 'ExterQual', 'Exterior1st', 'ExterCond', 'Foundation',
    'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars',
    'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'TotalBaths', 'FullBath', 'TotRmsAbvGrd',
    'YrSold', 'MoSold', 'SaleType', 'SaleCondition'
]

category_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler1', PowerTransformer()),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_power_transform_columns),
    ('cat', category_pipeline, category_feature_columns)
], remainder='drop')

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('reg', RegressorSwitcher())
])


models_for_stacking_l0 = [
    ('svr', SVR(C=0.75, kernel='rbf')),
    ('ridge', Ridge(alpha=1.75)),
    ('lasso', Lasso(alpha=0.25)),
    ('knr', KNeighborsRegressor(algorithm='brute', n_neighbors=9, p=1.0, weights='distance')),
    ('dt', DecisionTreeRegressor(max_depth=6)),
]


models_for_voting = [
    ('svr', SVR(C=0.75, kernel='rbf')),
    ('ridge', Ridge(alpha=1.75)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=16)),
    ('xgb', XGBRegressor(n_estimators=275, max_depth=5, learning_rate=0.08)),
    ('lgbm', LGBMRegressor(n_estimators=120, boosting_type='gbdt', learning_rate=0.05)),
    ('catb', CatBoostRegressor()),
    ('stack', StackingRegressor(estimators=models_for_stacking_l0))
]

model_weights = [
    # 3 levels of weights, 3, 2, 1
    2, # SVR
    2, # Ridge -
    1, # RF - lowest weight
    3, # XGB
    1, #LGBM
    3, #CatB
    3, #stack
]

model_weights = [i / sum(model_weights) for i in model_weights]

parameters = [
    {
        'reg__estimator': [VotingRegressor(estimators=models_for_voting, weights=model_weights)],
    },
]

X_train = train_df.drop(['SalePrice', 'SalePriceln'], axis=1)
y_train = train_df['SalePriceln']

gscv = GridSearchCV(model_pipeline, param_grid=parameters, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1, verbose=5)
gscv.fit(X_train, y_train)
print('Best Params: ', gscv.best_params_)
print('Best RMSLE: ', -1 * gscv.best_score_)

cvresults = pd.DataFrame(gscv.cv_results_).reset_index()
cvresults['model'] = cvresults['param_reg__estimator'].astype('str')
cvresults['model'] = cvresults['param_reg__estimator'].astype('str')
cvresults['mean_test_score'] *= -1
#cvresults.to_csv('cvresults 8 Multi model.csv')

summary_results = cvresults[['model','mean_test_score']].groupby('model').min().reset_index()
summary_results.sort_values(by='mean_test_score',ascending=True, inplace=True)
print(summary_results)


y_pred_final = gscv.predict(test_df)
y_pred_final_exp = np.expm1(y_pred_final)

submission = pd.DataFrame({'Id': test_Id, 'SalePrice': y_pred_final_exp})
filename = 'Submission 8 - Tuned Ensemble Stacking+Voting.csv'
submission.to_csv(filename, index=False)
print('\nSubmission file created: ', filename)









