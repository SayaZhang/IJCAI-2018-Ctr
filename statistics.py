# -*- coding: utf-8 -*-
"""
Created on Thur Mar 1 15:16:39 2018

@author: Sandra
"""
import pandas as pd
import matplotlib.pyplot as plt

# 统计
class statistic(object):
    
    def __init__(self):
        self.path = './data/'
        self.multy_feature_list = ['item_category_list','item_property_list','predict_category_property']
    
    # 读入数据
    def load_data(self):
        
        # 训练集
        self.train = pd.read_table(self.path+'round1_ijcai_18_train_20180301.txt',encoding='utf8',delim_whitespace=True)
        #self.train = pd.read_table(self.path+'sample.txt',encoding='utf8',delim_whitespace=True)
        self.train['isTrain'] = 1
    
        # 测试集
        self.test = pd.read_table(self.path+'round1_ijcai_18_test_a_20180301.txt',encoding='utf8',delim_whitespace=True)
        self.test['isTrain'] = 0
        
        # 连接
        self.df = pd.concat([self.train, self.test]).reindex()
        print("========> Load Data Success!")    
    
    # 1.数量
    def getCount(self):
        
        print(len(self.train))
        print(len(self.train.dropna(axis = 0)))
        print(len(self.test))
        print(len(self.test.dropna(axis = 0)))
        
    # 2.查看特征数量与分布
    def getSingleCategory(self):
        
        useless_feature = ['instance_id','isTrain']
        for x in self.df.columns:
            if (x not in self.multy_feature_list) or (x not in useless_feature):
                if x == 'is_trade':
                    freq_train = self.train[x].value_counts()
                    freq_train.to_csv('output/'+x+'.csv')
                else:
                    freq_train = self.train[x].value_counts()
                    freq_test = self.test[x].value_counts()
                    freq = pd.concat([freq_train,freq_test],axis=1)
                    freq.to_csv('output/'+x+'.csv')
                print("========> " + x + " freq Success!")
            #break
    
    def getDoubleFeature(self):
        
        double_feature = ['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']
        for x in double_feature:
            
            ser = self.df[x]
            #ser.index = self.df['instance_id']
            
            cats = pd.cut(ser[ser != -1], 10, labels=[1,2,3,4,5,6,7,8,9,10])
            #print(cats)
            #print(ser[ser != -1])
            #print(ser[ser == -1])
            ser = pd.concat([cats, ser[ser == -1]]).astype('int')
            #print(ser)
            #print(self.df[x])
            #break
            self.df[x+'_dispersed'] = ser
        print(self.df.loc[88064,:])
        
if __name__ =='__main__':
    
    s = statistic()
    s.load_data()
    s.getDoubleFeature()