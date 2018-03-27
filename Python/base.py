# -*- coding: utf-8 -*-
"""
Created on Thur Mar 1 15:16:39 2018

@author: Sandra
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
#import xgboost as xgb

# 读入数据
def load_data():
    
    path = './data/'
    
    # 训练集
    #train = pd.read_table(path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
    train = pd.read_table(path+'sample.txt',encoding='utf8',delim_whitespace=True)
    train['isTrain'] = 1
    train = train.dropna()
    '''
    # 测试集
    test = pd.read_table(path+'round1_ijcai_18_test_a_20180301.txt',encoding='utf8',delim_whitespace=True)
    test['isTrain'] = 0
    
    # 连接
    df = pd.concat([train,test])    
    print("========> Load Data Success!")
    return df
    '''
    return train

# one-hot编码处理   
def oneHot():
    
    df = load_data()
    dropCount = len(df) * 0.05    
    
    """
    特征：类别、属性列表 ONE-HOT

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
    特征：类别 ONE-HOT

    """  
    df = pd.DataFrame(m)
    category = df.loc[:,['instance_id','item_city_id','user_gender_id','context_page_id']]
    category.loc[:,['item_city_id','user_gender_id','context_page_id']] = category.loc[:,['item_city_id','user_gender_id','context_page_id']].astype('str')
    dfCategory = pd.get_dummies(category)
    df = pd.merge(df,dfCategory,on='instance_id')
    df = df.fillna(0)
    
    print("========> One Hot Success!")
    return df

# 训练
def train(): 
    
    df = oneHot()
    
    # init data set
    df_train = df[df['isTrain'] == 1]
    df_test = df[df['isTrain'] == 0] 
    
    # init feature
    feature=[x for x in df.columns if x not in ['instance_id','is_trade','item_id','user_id','context_id','isTrain','user_occupation_id','shop_id','item_brand_id','item_property_list','item_category_list','context_timestamp','predict_category_property','item_city_id','user_gender_id','context_page_id']]
    
    # preprocessing label
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['is_trade'].values))
    df_train['label'] = lbl.transform(list(df_train['is_trade'].values)) 
    
    # init model param
    num_class = df_train['label'].max()+1    
    params = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class':num_class,
        'silent' : 1
    }
    
    #print(feature)
    print(df_train.loc[1,feature]) 
    
    
train()
'''
# 读取数据   
path='../data/'
df=pd.read_csv(path+u'train-ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'train-ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'test-b-evaluation_public.csv')

# 读取规则数据
feature_path = 'output/'
#df_time = pd.read_csv(feature_path+u'shop-time-proba-process.csv')
f=open("time_pro", 'rb')
time_pro=pickle.load(f)
f.close()
df_wifi_once = pd.read_csv(feature_path+u'wifi-shop-once-filter.csv')

# 连表预处理
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
df['time_stamp']=pd.to_datetime(df['time_stamp'])
train=pd.concat([df,test])
mall_list=list(set(list(shop.mall_id)))

# 分开商场
result=pd.DataFrame()
for mall in mall_list:

    # init
    train1=train[train.mall_id==mall].reset_index(drop=True)       
    l=[]
    wifi_dict = {}

    # 处理wifi & create wifi dict
    for index,row in train1.iterrows():
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:

            # 添加wifi列
            row[i[0]]=int(i[1])

            # create wifi dict
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(row)    
    
    # create delete wifi list
    delate_wifi=[]
    for i in wifi_dict:
        if wifi_dict[i]<20:
            delate_wifi.append(i)

    # 过滤出现少于20次的wifi
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)

    # init data set
    train1=pd.DataFrame(m)
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]

    # preprocessing label
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    #print(df_train['label'].head())
    
    # init model param
    num_class=df_train['label'].max()+1    
    params = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'missing': -999,
        'num_class':num_class,
        'silent' : 1
    }

    # init feature
    # 只用经纬度和wifi

    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    num_rounds=60
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    
    # 输出
    b = model.predict(xgbtest)
    #df_test_n = pd.DataFrame(df_test_n, columns=[lbl.inverse_transform(i) for i in range(df_test_n.shape[1])],index=df_test['row_id'])
    
    # 加入时间概率分布
    time_now=[str(i).split(" ")[1] for i in df_test['time_stamp']]
    classes_list = [lbl.inverse_transform(i) for i in range(df_test_n.shape[1])]
    l=[list(i) for i in b]
    for i in range(0, len(l)):
        for j in range(0, len(l[i])):
            l[i][j]*=1
            l[i][j]*=time_pro[classes_list[j]][time_now[i]]
    a=[classes_list[i.index(max(i))] for i in l]
    del l, time_now, b

    # 返回结果
    temp_result=[]
    row = list(df_test['row_id'])
    for i in range(len(a)):
        temp_result.append([row[i],a[i]])
    temp_result
    # 假如wifi-shop强关联

    # concat result

    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'sub.csv',index=False)
    break

print('=====> sunccess')
'''