#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:01:47 2018

@author: hardyhe
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

path = '../Data/'

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

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
    
    # 对时间进行偏移
    df['time_real'] = df['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    df['date_real'] = df['time_real'].apply(lambda x: str(x).split(' ')[0] + ' 00:00:00')
    dates = list(pd.to_datetime(df['date_real']).drop_duplicates().sort_values()) # 18 19 20 21 22 23 24 | 25
    print(dates)
    
    count = 0
    data = []
    for item in df.groupby('item_id'):
        
        item_cvr_dict = {}
        print(count)
        
        # mean cvr of train
        train_len = len(item[1][item[1]['isTrain'] == 1])
        cvr = train_len > 0 and len(item[1][(item[1]['isTrain'] == 1) & (item[1]['is_trade'] == 1)])/train_len or 0
        # mean cvr
        item_cvr_dict['mean'] = cvr
        # first day cvr
        item_cvr_dict[str(dates[0])] = cvr/7       
        # 从第二天开始，得到前一天的cvr 存在dict中
        for i in range(1,len(dates)):
            last_day = dates[i] - datetime.timedelta(days=1) 
            last_day_item = item[1][item[1]['date_real'] == str(last_day)]
            last_day_item_len = len(last_day_item)
            last_day_cvr = last_day_item_len > 0 and len(last_day_item[last_day_item['is_trade'] == 1])/last_day_item_len or 0
            item_cvr_dict[str(dates[i])] = last_day_cvr      
        
        for index,row in item[1].iterrows():
            data.append([row['instance_id'], row['isTrain'], item_cvr_dict['mean'], item_cvr_dict[row['date_real']]])
        count += 1
        #break
        
    data = pd.DataFrame(data, columns=['instance_id', 'isTrain', 'cvr', 'last_day_cvr'])
    data.to_csv(path + 'features/' + name + '_history.csv',index=None)
    
def date_stat():
    df = load_data()
    df['time'] = pd.to_datetime(df.context_timestamp, unit='s')
    df['time_real'] = df['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    
    df['date'] = df['time_real'].apply(lambda x: str(x).split(":")[0] + ':00:00')
    df[df['isTrain'] == 1]['date'].value_counts().sort_index().to_csv('../Stat_output/train_time.csv')
    df[df['isTrain'] == 0]['date'].value_counts().sort_index().to_csv('../Stat_output/test_time.csv')

item_history_feature('item_id')