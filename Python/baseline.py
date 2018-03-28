# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import datetime
import pandas as pd
import gbdt_lr_train 


def convert_data(data):
    data['time'] = pd.to_datetime(data.context_timestamp, unit='s')
    # 对时间进行偏移
    data['time_real'] = data['time'].apply(lambda x: x + datetime.timedelta(hours=8))
    data['day'] = data['time_real'].apply(lambda x: int(str(x)[8:10]))
    data['hour'] = data['time_real'].apply(lambda x: int(str(x)[11:13]))
    # user_id join day
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    # user_id join day hour
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])

    return data


if __name__ == "__main__":
    online = False# 这里用来标记是 线下验证 还是 在线提交
    path = '../Data/'

    data = pd.read_csv(path + 'round1_ijcai_18_train_20180301.txt', sep=' ')
    print('load data success!')
    item_history_cvr = pd.read_csv(path + 'features/item_id_history.csv')
    print('load history data success!')    
    data = pd.merge(data,item_history_cvr[item_history_cvr['isTrain'] == 1],on='instance_id')
    print('merge history data success!') 
    
    data.drop_duplicates(inplace=True)
    data = convert_data(data)

    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    elif online == True:
        train = data.copy()
        test = pd.read_csv('../data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
        print(len(test))
        test = pd.merge(test,item_history_cvr[item_history_cvr['isTrain'] == 0],on='instance_id')
        test = convert_data(test)

    features = ['item_id','item_brand_id','item_city_id','shop_id','item_price_level', 'item_sales_level','item_collected_level', 'item_pv_level', 
                     'user_gender_id','user_occupation_id','user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                     'context_page_id', 'hour', 'shop_review_num_level', 'shop_star_level','shop_review_positive_rate', 
                     'shop_score_service', 'shop_score_delivery', 'shop_score_description','cvr','last_day_cvr']
    
    target = ['is_trade']

    if online == False:
        gbdt_lr_train.gbdt_lr_train(train[features],train[target],test[features],test[target])
    else:
        gbdt_lr_train.predict(train[features],train[target],test,features,'with_history_cvr')

