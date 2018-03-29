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

def gbdt_lr_train(train,test,gbdt_features,lr_features,target,name,isOnline):

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=20, max_depth=3, verbose=0, max_features=0.3)
    #n_estimators=20, max_depth=3, verbose=0, max_features=0.5

    # 训练学习
    gbdt.fit(train[gbdt_features], train[target])

    # 预测及AUC评测
    if isOnline == False:
        y_pred_gbdt = gbdt.predict_proba(test[gbdt_features])[:, 1]
        gbdt_test_log_loss = log_loss(test[target], y_pred_gbdt)
        print('gbdt log_loss: %.5f' % gbdt_test_log_loss)
    else:
        y_pred_gbdt = gbdt.predict_proba(train[gbdt_features].tail(57562))[:, 1]
        gbdt_test_log_loss = log_loss(train[target].tail(57562), y_pred_gbdt)
        print('gbdt log_loss: %.5f' % gbdt_test_log_loss)

    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(train[gbdt_features])[:,:,0]
    X_test_leaves = gbdt.apply(test[gbdt_features])[:,:,0]

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

    # 定义LR模型
    lr = LogisticRegression()
    
    # lr对gbdt特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], train[target])
    
    # 预测及AUC评测
    if isOnline == False:
        y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
        gbdt_lr_test_log_loss1 = log_loss(test[target], y_pred_gbdtlr1)
        print('基于GBDT特征编码后的LR log_loss: %.5f' % gbdt_lr_test_log_loss1)
    else:
        print('Online')

    # 定义LR模型
    lr = LogisticRegression()
    
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], train[lr_features]])
    X_test_ext = hstack([X_trans[train_rows:, :], test[lr_features]])
    
    print("gbdt output",X_trans[:train_rows, :].shape)
    print("input",train[lr_features].shape)
    print(X_train_ext.shape)
    
    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, train[target])

    # 预测及AUC评测
    if isOnline == False:
        y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
        gbdt_lr_test_log_loss2 = log_loss(test[target], y_pred_gbdtlr2)
        print('基于组合特征的LR log_loss: %.5f' % gbdt_lr_test_log_loss2)
    else:
        print('Online')
        
        test['predicted_score'] = lr.predict_proba(X_test_ext)[:, 1]
        print(test['predicted_score'].head(5))
        print(len(test))
        test[['instance_id', 'predicted_score']].to_csv('../baseline_' + name +'.csv', index=False,sep=' ')#保存在线提交结果
        print('Saved result success!')
    