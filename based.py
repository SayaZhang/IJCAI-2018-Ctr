# -*- coding: utf-8 -*-
"""
Created on Thur Mar 1 15:16:39 2018

@author: Sandra
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, KFold 
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
#import xgboost as xgb


"""
特征列表
"""
# 1. 类目列表特征
listItem = ['item_category_list','item_property_list']

# 2. 类别特征
singleIntItem = ['item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','item_brand_id']
singleIntUser = ['user_gender_id','user_age_level','user_occupation_id','user_star_level']
singleIntContext = ['context_page_id']
singleIntShop = ['shop_review_num_level','shop_star_level']
singleIntFeature = singleIntItem + singleIntUser + singleIntContext + singleIntShop

# 3. 连续型特征
singleDoubleShop = ['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']
singleDoubleShopDispersed = ['shop_review_positive_rate_dispersed','shop_score_service_dispersed','shop_score_delivery_dispersed','shop_score_description_dispersed']

# 4. ID列表
idList = ['instance_id','item_id','user_id','context_id','shop_id' ]

# 5. 目前还未用到的特征
unsureList = ['context_timestamp','predict_category_property']

# 5 train label标记
label = ['isTrain', 'is_trade']

"""
读入数据
"""
def load_data():
    
    path = './data/'
    
    # 训练集
    #train = pd.read_table(path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
    train = pd.read_table(path+'sample.txt',encoding='utf8',delim_whitespace=True)
    train['isTrain'] = 1
    train = train.dropna()

    # 测试集
    #test = pd.read_table(path+'round1_ijcai_18_test_a_20180301.txt',encoding='utf8',delim_whitespace=True)
    test = pd.read_table(path+'test_sample.txt',encoding='utf8',delim_whitespace=True)
    test['isTrain'] = 0
    
    # 连接
    df = pd.concat([train,test]) 
    print("========> Load Data Success!")
    return df

    #return train

"""
one-hot编码处理 
"""  
def oneHot():
    
    df = load_data()
    dropCount = len(df) * 0.05    
    
    """
    1. 特征: 类别、属性列表 ONE-HOT
    item_category_list
    item_property_list 

    """    
    l=[]
    item_category_dict = {}
    item_property_dict = {}
    
    # item_category & create item_category dict
    for index,row in df.iterrows():
        
        item_category_list = [x for x in row['item_category_list'].split(';')]
        item_property_list = [x for x in row['item_property_list'].split(';')]
        
        #item_category_list
        for x in item_category_list:
            # 添加item_category列
            row['item_category_'+x] = 1

            # create wifi dict
            if x not in item_category_dict:
                item_category_dict['item_category_'+x] = 1
            else:
                item_category_dict['item_category_'+x] += 1
        
        #item_property_list
        for x in item_property_list:
            # 添加item_category列
            row['item_property_'+x] = 1

            # create wifi dict
            if x not in item_property_dict:
                item_property_dict['item_property_'+x] = 1
            else:
                item_property_dict['item_property_'+x] += 1
                
        l.append(row)

    # create delete item category list
    delete_item_category_list = []
    for i in item_category_dict:
        if item_category_dict[i]< dropCount:
            delete_item_category_list.append(i)
    
    # create delete item property list
    delete_item_property_list = []
    for i in item_property_dict:
        if item_property_dict[i]< dropCount:
            delete_item_property_list.append(i)

    # 过滤
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delete_item_category_list:
                new[n]=row[n]
            if n not in delete_item_property_list:
                new[n]=row[n]
        m.append(new)
    
    """
    2. 特征: 类别 ONE-HOT
    item_city_id 
    item_price_level 
    item_sales_level 
    item_collected_level 
    item_pv_level 
    item_brand_id
    user_gender_id 
    user_age_level 
    user_occupation_id 
    user_star_level 
    context_page_id
    shop_review_num_level
    shop_star_level
    """  
    df = pd.DataFrame(m)
    singleIntFeatureList = singleIntFeature + ['instance_id']
    category = df.loc[:,singleIntFeatureList]
    category.loc[:,singleIntFeature] = category.loc[:,singleIntFeature].astype('str')
    dfCategory = pd.get_dummies(category)
    df = pd.merge(df,dfCategory,on='instance_id')
    
    """
    3. 特征: 浮点数 离散化+OneHot
    shop_review_positive_rate
    shop_score_service 
    shop_score_delivery 
    shop_score_description 
    """
    for x in singleDoubleShop:            
        ser = df[x]            
        cats = pd.cut(ser[ser != -1], 10, labels=[1,2,3,4,5,6,7,8,9,10])
        ser = pd.concat([cats, ser[ser == -1]]).astype('int')
        df[x+'_dispersed'] = ser
    
    singleDoubleShopDispersedList = singleDoubleShopDispersed + ['instance_id']
    category = df.loc[:,singleDoubleShopDispersedList]
    category.loc[:,singleDoubleShopDispersed] = category.loc[:,singleDoubleShopDispersed].astype('str')
    dfCategory = pd.get_dummies(category)
    df = pd.merge(df,dfCategory,on='instance_id')        
    df = df.fillna(0)
    
    print("========> One Hot Success!")
    return df

# log likelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess


"""
训练
"""
def train(): 
    
    df = oneHot()
    print('========> start train!')
    
    # init data set
    df_train = df[df['isTrain'] == 1]
    df_test = df[df['isTrain'] == 0] 
    
    # init feature
    UselessFeature = idList + singleDoubleShopDispersed + singleDoubleShop + singleIntFeature + listItem + unsureList + label
    feature=[x for x in df.columns if x not in UselessFeature]
    df.loc[:,feature].to_csv('feature.csv',index=None)
    
    # 10-fold
    kf = KFold(n=len(df_train), n_folds=10, shuffle=False) 
    kfcount = 0
    for train_index, test_index in kf:
        
        print("\n===========" + str(kfcount) + "===========")
        
        # prepare train data
        X_train, y_train = df_train.loc[train_index, feature], df_train.loc[train_index, 'is_trade']
        X_test, y_test = df_train.loc[test_index, feature], df_train.loc[test_index, 'is_trade']
       
        # init and train model
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, y_train)
        
        # predict train
        y_train_pred = clf.predict(X_train)
        
        # evaluate train logLoss
        one_hot = OneHotEncoder(n_values=2, sparse=False)
        y_train = one_hot.fit_transform(np.array([[x] for x in y_train]))
        y_train_pred = one_hot.fit_transform(np.array([[x] for x in y_train_pred]))
        logLoss_train = log_loss(y_train, y_train_pred)
        print("train:", logLoss_train)
        
        # predict validation data
        y_validate_pred = clf.predict(X_test)
        
        # evaluate validate logLoss
        y_validate = one_hot.fit_transform(np.array([[x] for x in y_test]))
        y_validate_pred = one_hot.fit_transform(np.array([[x] for x in y_validate_pred]))
        logLoss_validate = log_loss(y_validate, y_validate_pred)
        print("train:", logLoss_validate)
        
        kfcount += 1
        #break
    
    #df_test['predict'] = [x[0] for x in result]
    #df_test[['instance_id','predict']].astype('str').to_csv('result.csv',index=None)

train()
