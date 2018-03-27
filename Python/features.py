#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:01:47 2018

@author: hardyhe
"""

import pandas as pd
import datetime

path = '../Data/'
def load_data():
    
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
    print("========> Load Data Success!")
    return df
    
def item_history_feature(name):
    
    df = load_data()
    df = df[['instance_id', name, 'context_timestamp', 'is_trade','isTrain']]   
    df['time'] = pd.to_datetime(df.context_timestamp, unit='s')
    
    count = 0
    data = []
    for item in df[['instance_id', name, 'time', 'is_trade','isTrain']].groupby('item_id'):
        print(count)
        
        # train 中 cvr
        train_len = len(item[1][item[1]['isTrain'] == 1])
        cvr = train_len > 0 and len(item[1][(item[1]['isTrain'] == 1) & (item[1]['is_trade'] == 1)])/train_len or 0
        
        for index,row in item[1].iterrows():
            
            last_day = row['time'] - datetime.timedelta(days=1)            
            last_day_item = item[1][item[1]['time'] < last_day]
            last_day_item_len = len(last_day_item)           
            last_day_cvr = last_day_item_len > 0 and len(last_day_item[last_day_item['is_trade'] == 1])/last_day_item_len or 0
            
            data.append([row['instance_id'], cvr, last_day_cvr])
            
        count += 1
        
        #break
    
    data = pd.DataFrame(data, columns=['instance_id','cvr','last_day_cvr'])
    data.to_csv(path + 'features/' + name + '_history.csv',index=None)
    
def date_stat():
    df = load_data()
    df['time'] = pd.to_datetime(df.context_timestamp, unit='s')
    df['date'] = df['time'].apply(lambda x: str(x).split(" ")[0])
    print(df[df['isTrain'] == 1]['date'].value_counts)
    print(df[df['isTrain'] == 0]['date'].value_counts)

item_history_feature('item_id')