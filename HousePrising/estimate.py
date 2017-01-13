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

  def execute(self):
    X_train = self.df.drop("SalePrice",axis=1).iloc[:,:-1].values
    # print X
    y_train = self.df["SalePrice"].values

    X_test = self.test_df.iloc[:,:-1].values
    # print X
    # y_train = self.df["SalePrice"].values
    # print y.size
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        # test_size=0.4,
                                                        # random_state=1)
    forest = RandomForestRegressor(criterion='mse',
                                   random_state=1,
                                   n_jobs=-1)
    #gridsearchしたいハイパーパラメータの入力
    params =  {
    'n_estimators' : [600, 700, 800]
    }
    forest = GridSearchCV(estimator=forest, param_grid=params)
    forest.fit(X_train, y_train)
    #gridsearchによる最適なハイパーパラメータを出力
    print (forest.best_params_)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    print y_test_pred
    y_test_pred.to_csv("submission.csv")

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
    # self.all_df = pd.concat((self.df.loc[:,"MSSubClass":"SaleCondition"], self.test_df.loc[:,"MSSubClass":"SaleCondition"]))

  def preProcessing(self):
    #　欠損値補完のインスタンスを生成
    imr = Imputer(missing_values="Nan", strategy="most_frequent", axis=0)
    self.all_df = pd.get_dummies(self.all_df)
    # imr = imr.fit(self.all_df.dropna())
    self.df = pd.get_dummies(self.df)
    self.df = self.df.reindex(columns = self.all_df.columns, fill_value=0)
    self.test_df = pd.get_dummies(self.test_df)
    self.test_df = self.test_df.reindex(columns = self.all_df.columns, fill_value=0)


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
