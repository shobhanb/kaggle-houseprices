import numpy as np
import pandas as pd
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
pandas.set_option('display.max_rows', 100)
pandas.set_option('display.max_columns', 100)
sns.set()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_Id = test_df['Id']
super_df = train_df.append(test_df)
combine_df = [train_df, test_df, super_df]

#print(super_df.info())
#print(super_df.isnull().sum().sort_values(ascending=False))
#print(super_df.nunique())

def ZScoreScale(*column_names):
    for colname in column_names:
        mean = super_df[colname].dropna().mean()
        std = super_df[colname].dropna().std()
        for df in combine_df:
            df[colname] = (df[colname] - mean)/std

def numericcolinfo(colname, plots=True):
    print('\nColumn info for: ', colname)
    print(super_df[colname].describe())
    print('NUnique: ', super_df[colname].nunique())
    print('Isnull: ', super_df[colname].isnull().sum())
    print('IsZero: ', (super_df[colname] == 0).sum())
    if plots == True:
        ax1 = sns.boxplot(y=colname, data=super_df)
        ax2 = sns.countplot(colname, data=super_df)
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
#print(train_df['SalePrice'].describe(percentiles=[0.995]))
SP_clip_upper = 500000
for df in [train_df, super_df]:
    df['SalePrice'] = df['SalePrice'].clip(upper=SP_clip_upper)
    df['SalePriceln'] = np.log1p(df['SalePrice'])


numeric_feature_columns = []
category_feature_columns = []



#OverallQual
#numericcolinfo('OverallQual')
ZScoreScale('OverallQual')
numeric_feature_columns.append('OverallQual')


#GrLivArea
#numericcolinfo('GrLivArea')
ZScoreScale('GrLivArea')
# Clip to 3 std deviations (after scaling)
for df in combine_df:
    df['GrLivArea'] = df['GrLivArea'].clip(upper=3)
numeric_feature_columns.append('GrLivArea')


# All the Garage variables

# GarageType
#categorycolinfo('GarageType')

# Having a garage or not, Garage Area, and Garage # of cars seem to be the most usable

for df in combine_df:
    df['noGarage'] = df['GarageArea']==0
numeric_feature_columns.append('noGarage')
#numericcolinfo('noGarage')

# GarageArea

# Found 1 NA in GarageArea in test_df. Let's fill it based on the Garage Type median
#print(test_df.loc[test_df['GarageArea'].isnull(),'GarageType'])
# Found it to be Detached garage type. Use that to fill the NA
fill_GarageArea = super_df.loc[super_df['GarageType']=='Detchd', 'GarageArea'].dropna().median()

for df in combine_df:
    df.loc[df['GarageArea'].isnull(), 'GarageArea'] = fill_GarageArea


ZScoreScale('GarageArea')
# Let's clip to 3 std deviations
for df in combine_df:
    df['GarageArea'] = df['GarageArea'].clip(upper=3)

numeric_feature_columns.append('GarageArea')

# GarageCars
# Clip to 4 Garages (before scaling) and fill one NA with 0
for df in combine_df:
    df['GarageCars'] = df['GarageCars'].clip(lower=0, upper=3).fillna(value=0)
ZScoreScale('GarageCars')
#numericcolinfo('GarageCars',False)
numeric_feature_columns.append('GarageCars')



# TotalBsmtSF
# Obviously a high correlation. Will also look at all the other Basement fields here
#basement_cat_fields = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
#basement_num_fields = ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
#for col in basement_cat_fields:
#    categorycolinfo(col)
#for col in basement_num_fields:
#    numericcolinfo(col, False)

# Found 1 null in Test. ALl other Basement fields were Nan for it, so filled TotalSF with 0
#print('Null BsmtSF: ',test_df['TotalBsmtSF'].isnull().sum())
#print(test_df.loc[test_df['TotalBsmtSF'].isnull(), basement_cat_fields])

for df in combine_df:
    df['TotalBsmtSF'].fillna(value=0, inplace=True)
    df['hasBasement'] = df['TotalBsmtSF'] == 0

ZScoreScale('TotalBsmtSF')
numeric_feature_columns.append('hasBasement')
numeric_feature_columns.append('TotalBsmtSF')


# 1stFlr & 2ndFlr SF
#numericcolinfo('1stFlrSF')
ZScoreScale('1stFlrSF')
numeric_feature_columns.append('1stFlrSF')

#numericcolinfo('2ndFlrSF')
for df in combine_df:
    df['has2ndFlr'] = df['2ndFlrSF'] == 0
numeric_feature_columns.append('has2ndFlr')

ZScoreScale('2ndFlrSF')
numeric_feature_columns.append('2ndFlrSF')


# Bathrooms
for df in combine_df:
    df['TotalBaths'] = df['FullBath'] + df['HalfBath'] + df['BsmtFullBath'] + df['BsmtHalfBath']
    df['TotalBaths'] = df['TotalBaths'].fillna(value=0).clip(upper=4)

ZScoreScale('TotalBaths')
numeric_feature_columns.append('TotalBaths')

# Rooms
#numericcolinfo('TotRmsAbvGrd')

for df in combine_df:
    df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].clip(lower=3,upper=10)

ZScoreScale('TotRmsAbvGrd')
numeric_feature_columns.append('TotRmsAbvGrd')


# Yr Built / Remod

#sns.jointplot('YearRemodAdd', 'YearBuilt', data=super_df)
#plt.show()
ZScoreScale('YearBuilt')
numeric_feature_columns.append('YearBuilt')
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



#corrmat = train_df.corr().abs()['SalePrice'].sort_values()
#print(corrmat)

#print(numeric_feature_columns)
#print(category_feature_columns)


#for col in numeric_feature_columns:
#    plt.clf()
#    sns.distplot(train_df[col])
#    plt.show()



# Let's start with a simple Regression Model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

X = train_df[numeric_feature_columns]
y = train_df['SalePriceln']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)
score_train = -1 * cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')

y_pred = lr.predict(X_test)
score_test = np.sqrt(mean_squared_error(y_test, y_pred))

coeffs = pd.DataFrame(lr.coef_, index=numeric_feature_columns, columns=['Coeffs'])
print('Coefficients: \n', coeffs.sort_values(by='Coeffs'))
print('Intercept: ', lr.intercept_)
print('RMSLE on Train: ', score_train.mean())
print('RMSLE Score on Test : ', score_test)
print('R2 on Test: ', r2_score(y_test, y_pred))


y_pred_final = lr.predict(test_df[numeric_feature_columns])
y_pred_final_exp = np.expm1(y_pred_final)

submission = pd.DataFrame({'Id':test_Id, 'SalePrice':y_pred_final_exp})
filename = 'Submission 1 - Basic Linear Regression.csv'
submission.to_csv(filename, index=False)
print('\nSubmission file created: ', filename)
