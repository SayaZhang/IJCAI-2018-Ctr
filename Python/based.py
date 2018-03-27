# -*- coding: utf-8 -*-
"""
Created on Thur Mar 1 15:16:39 2018

@author: Sandra
"""
import time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
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
unsureList = ['predict_category_property']

# 5 train label标记
label = ['isTrain', 'is_trade']

# time
timeFeature = ['context_timestamp','time']

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data

"""
读入数据
"""
def load_data():
    
    path = './data/'
    
    # 训练集
    train = pd.read_table(path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
    #train = pd.read_table(path+'sample.txt',encoding='utf8',delim_whitespace=True)
    train.drop_duplicates(inplace=True)
    train['isTrain'] = 1
    train = train.dropna()

    # 测试集
    test = pd.read_table(path+'round1_ijcai_18_test_a_20180301.txt',encoding='utf8',delim_whitespace=True)
    #test = pd.read_table(path+'test_sample.txt',encoding='utf8',delim_whitespace=True)
    test['isTrain'] = 0
    
    # 连接
    df = pd.concat([train,test]) 
    df = convert_data(df)
    print("========> Load Data Success!")
    return df

"""
one-hot编码处理 
"""  
def oneHot():
    
    df = load_data()
    #dropCount = 20    
    
    """
    1. 特征: 类别、属性列表 ONE-HOT
    item_category_list
    item_property_list 

        
    l=[]
    item_category_dict = {}
    item_property_dict = {}
    
    # item_category & create item_category dict
    for index,row in df.iterrows():
        
        print(index)
        
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
    count = 0
    print("============================")
    for row in l:
        print(count)
        new={}
        for n in row.keys():
            if n not in delete_item_category_list:
                new[n]=row[n]
            if n not in delete_item_property_list:
                new[n]=row[n]
        m.append(new)
        count += 1
    print("  =========> Part 1!")
    """
    
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
    #df = pd.DataFrame(m)
    singleIntFeatureList = singleIntFeature + ['instance_id']
    category = df.loc[:,singleIntFeatureList]
    category.loc[:,singleIntFeature] = category.loc[:,singleIntFeature].astype('str')
    dfCategory = pd.get_dummies(category)
    df = pd.merge(df,dfCategory,on='instance_id')
    print("  =========> Part 2!")
    
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
    print("  =========> Part 3!")        
    
    df = df.fillna(0)    
    print("========> One Hot Success!")
    return df

"""
训练
"""
def train(): 
    
    df = oneHot()
    print('========> start train!')
    
    # init dataset
    df_train = df[df['isTrain'] == 1]
    df_test = df[df['isTrain'] == 0] 
    
    # init feature
    UselessFeature = idList + singleDoubleShopDispersed + singleDoubleShop + singleIntFeature + listItem + unsureList + timeFeature + label
    feature=[x for x in df.columns if x not in UselessFeature]
    target = ['is_trade']
    #df.loc[:,feature].to_csv('feature.csv',index=None)
    
    online = False# 这里用来标记是 线下验证 还是 在线提交
    if online == False:
        train = df_train.loc[df_train.day < 24]  # 18,19,20,21,22,23,24
        test = df_train.loc[df_train.day == 24]  # 暂时先使用第24天作为验证集
    elif online == True:
        train = df_train.copy()
        test = df_test  

    if online == False:

        clf = GradientBoostingClassifier(max_leaf_nodes=63, max_depth=7, n_estimators=80)
        clf.fit(train[feature], train[target])
        
        test['lgb_predict'] = clf.predict_proba(test[feature],)[:, 1]
        print(log_loss(test[target], test['lgb_predict']))
    else:
        
        clf = GradientBoostingClassifier(max_leaf_nodes=63, max_depth=7, n_estimators=80)
        clf.fit(train[feature], train[target])
        
        test['predicted_score'] = clf.predict_proba(test[feature])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline.csv', index=False,sep=' ')#保存在线提交结果  

train()
