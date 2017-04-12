#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
from sklearn.preprocessing import Imputer

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os
os.environ['PATH'] = os.environ['PATH'] + ';C:¥TDM-GCC-64¥bin'
import xgboost as xgb

class ExploratoryDataAnalysis:
  def __init__(self, df):
    self.df = df

  def execute(self):
    sns.set(style="whitegrid", context="notebook")
    sns.pairplot(self.df.dropna(), size = 3.5)
    # sns.pairplot(df.iloc[:,1:11].dropna(), size = 3.5)
    plt.show()

class RandomForest:
  def __init__(self, df, test_df):
    self.df = df
    self.test_df = test_df
    self.stdsc = StandardScaler()

  def execute(self):
    X_train = self.df.drop("SalePrice",axis=1).iloc[:,:-1].values
    # print X
    X_train = self.stdsc.fit_transform(X_train)
    y_train = self.df["SalePrice"].values

    X_test = self.test_df.drop("SalePrice",axis=1).iloc[:,:-1].values
    # print X
    X_test = self.stdsc.transform(X_test)
    # y_train = self.df["SalePrice"].values
    # print y.size


    # X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        # test_size=0.4,
                                                     # random_state=1)

    # forest = RandomForestRegressor(criterion='mse',
    #                                random_state=1,
    #                                n_jobs=-1)
    # #gridsearchしたいハイパーパラメータの入力
    # params =  {
    #             'n_estimators' : [690, 699, 700, 691,692,693,694,695,696,697,698]
    #             }
    # forest = GridSearchCV(estimator=forest, param_grid=params)
    # # print X_train.shape
    # # print X_test.shape

    # forest.fit(X_train, y_train)
    # #gridsearchによる最適なハイパーパラメータを出力
    # print (forest.best_params_)
    # # importances = forest.feature_importances_
    # # indices = np.argsort(importances)[::-1]
    # # for f in range(X_train.shape[1]):
    # #   print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
    # y_train_pred = forest.predict(X_train)
    # y_test_pred = forest.predict(X_test)

    # 線形regression
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)
    # y_test_pred = regressor.predict(X_test)

    #  dtrain = xgb.DMatrix(X_train, y_train)
    #  dtest = xgb.DMatrix(X_test, y_test)
    # params = {"max_depth":2, "eta":0.1}
    #  model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=10)
    #  model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()



    params = {'max_depth': [3, 5, 10], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 10, 100], 'subsample': [0.8, 0.85, 0.9, 0.95], 'colsample_bytree': [0.5, 1.0]}
    # モデルのインスタンス作成
    model_xgb = xgb.XGBRegressor()
    # 10-fold Cross Validationでパラメータ選定
    # model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
    model_xgb.fit(X_train, y_train)
    y_test_pred = model_xgb.predict(X_test)
    # cv = GridSearchCV(model_xgb, params, cv = 10, scoring= 'mean_squared_error', n_jobs =1)
    # cv.fit(X_train, y_train)
    # cv.fit(X_train, y_train)
    # y_test_pred = cv.predict(X_test)

    output_df = pd.DataFrame({"Id" : np.arange(1461, 2920, 1),"SalePrice" : y_test_pred})
    output_df.to_csv("submission.csv", index=None)


    # #平均二乗誤差
    # print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
    #                                          mean_squared_error(y_test, y_test_pred)))
    # #誤差の平均
    # print('residuals train: %.3f, test: %.3f' % (math.sqrt(mean_squared_error(y_train, y_train_pred)),
    #                                      math.sqrt(mean_squared_error(y_test, y_test_pred))))
    # #平均誤差率
    # print('residuals ratio train: %.6f, test: %.6f' % (np.average(abs(y_train_pred - y_train)/ y_train),
    #                                      np.average(abs(y_test_pred - y_test) / y_test)) )

    # #決定二乗誤差
    # print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
    #                                        r2_score(y_test, y_test_pred)))


class MachineLearning:
  def __init__(self, csv, test_csv):
    self.csv = csv
    self.test_csv = test_csv
    self.df = pd.read_csv(self.csv)
    self.test_df = pd.read_csv(self.test_csv)
    self.all_df = pd.concat((self.df, self.test_df) )
    # self.all_df.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
    #            'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
    #            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
    #           axis=1, inplace=True)
    # self.df.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
    #            'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
    #            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
    #           axis=1, inplace=True)
    # self.test_df.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
    #            'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
    #            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
    #           axis=1, inplace=True)
    # self.all_df = pd.concat((self.df.loc[:,"MSSubClass":"SaleCondition"], self.test_df.loc[:,"MSSubClass":"SaleCondition"]))

  def preProcessing(self):

    #　欠損値補完のインスタンスを生成
    # imr = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
    self.all_df = pd.get_dummies(self.all_df)
    # imr = imr.fit(self.all_df.dropna())
    self.df = pd.get_dummies(self.df)
    self.df = self.df.reindex(columns = self.all_df.columns, fill_value=0)
    self.test_df = pd.get_dummies(self.test_df)
    self.test_df = self.test_df.reindex(columns = self.all_df.columns, fill_value=0)
    # 欠損補完
    self.all_df = self.all_df.fillna(self.all_df.mean())
    self.df = self.df.fillna(self.df.mean())
    self.test_df = self.test_df.fillna(self.test_df.mean())




    # valid_cols=["SalePrice","平均価格","最大価格","最小価格","販売方法","階数","総専面積","販売戸数","平均専面","敷地面積","建築面積","延床面積","販売戸数"]

    # #同じ建物を削除
    # self.df = self.df.drop_duplicates(["建物名"])

    #間取り、駅名と管理会社用のダミー変数を作成
    # self.df = pd.get_dummies(self.df)

    #欠損値の削除
    # self.df.isnull().sum()
    # self.df = self.df.dropna()
    # self.df = imr.transform(self.df)
    self.df.to_csv("PreProcess.csv")


    # self.test_df = pd.get_dummies(self.test_df)

    #欠損値の削除
    # self.test_df.isnull().sum()
    # self.test_df = self.test_df.dropna()
    # self.test_df = imr.transform(self.test_df)
    self.test_df.to_csv("PreProcessTest.csv")

  def execEDA(self):
    eda = ExploratoryDataAnalysis(self.df)
    eda.execute()

  def randomForest(self):
    self.rf = RandomForest(self.df, self.test_df)
    self.rf.execute()

  # def precict(self, dat):
  #   self.rf.estimate()


ml = MachineLearning(csv="train.csv", test_csv="test.csv")
ml.preProcessing()
ml.randomForest()
