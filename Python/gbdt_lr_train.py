#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:20:03 2018

@author: hardyhe
"""

from scipy.sparse.construct import hstack
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.metrics import log_loss
import numpy as np

def gbdt_lr_train(X_train,y_train,X_test,y_test):

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=20, max_depth=3, verbose=0, max_features=0.5)
    #n_estimators=20, max_depth=3, verbose=0, max_features=0.5

    # 训练学习
    gbdt.fit(X_train, y_train)

    # 预测及AUC评测
    y_pred_gbdt = gbdt.predict_proba(X_test)[:, 1]
    gbdt_test_log_loss = log_loss(y_test, y_pred_gbdt)
    print('gbdt log_loss: %.5f' % gbdt_test_log_loss)

    # lr对原始特征样本模型训练
    lr = LogisticRegression()
    lr.fit(X_train, y_train)    # 预测及AUC评测
    y_pred_test = lr.predict_proba(X_test)[:, 1]
    lr_test_log_loss = log_loss(y_test, y_pred_test)
    print('基于原有特征的LR log_loss: %.5f' % lr_test_log_loss)

    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(X_train)[:,:,0]
    X_test_leaves = gbdt.apply(X_test)[:,:,0]

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

    # 定义LR模型
    lr = LogisticRegression()
    # lr对gbdt特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    gbdt_lr_test_log_loss1 = log_loss(y_test, y_pred_gbdtlr1)
    print('基于GBDT特征编码后的LR log_loss: %.5f' % gbdt_lr_test_log_loss1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    print(X_train_ext.shape)
    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    gbdt_lr_test_log_loss2 = log_loss(y_test, y_pred_gbdtlr2)
    print('基于组合特征的LR log_loss: %.5f' % gbdt_lr_test_log_loss2)

def predict(X_train,y_train,X_test,features,name):
    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=20, max_depth=3, verbose=0, max_features=0.5)

    # 训练学习
    gbdt.fit(X_train, y_train)
    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(X_train)[:,:,0]
    X_test_leaves = gbdt.apply(X_test[features])[:,:,0]

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

    # 定义LR模型
    lr = LogisticRegression()
    # lr对gbdt特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    X_test['predicted_score'] = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    print(X_test['predicted_score'].head(5))
    print(len(X_test))
    X_test[['instance_id', 'predicted_score']].to_csv('../baseline_' + name +'.csv', index=False,sep=' ')#保存在线提交结果