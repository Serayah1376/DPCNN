# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:09:47 2021

@author: 10983
"""

import pandas as pd
import torch.utils.data as torch_data
from nltk import word_tokenize#以空格形式进行分词
import torch
USE_GPU=True


class CodeDataset(torch_data.Dataset):
    def __init__(self, is_train_set=True,is_test_set=False):
        super().__init__()
        #定义label和0-103对应的字典（训练集中一共有104种标签）
        train_file = 'C:/Users/10983/py入门/GRUClassifier/train.pb'
        self.train_reader = pd.read_pickle(train_file)
        self.train_label = self.train_reader.loc[:, 'label']
        self.train_label_list = sorted(self.train_label.unique())
        
        
        if is_test_set:
            filename='C:/Users/10983/py入门/GRUClassifier/test.pb'
        elif is_train_set:
            filename='C:/Users/10983/py入门/GRUClassifier/train.pb'
        else:
            filename='C:/Users/10983/py入门/GRUClassifier/valid.pb'
        reader = pd.read_pickle(filename)
        self.codes_tmp = reader.loc[:, 'code'].values # 将代码列表放入codes中
        self.codes=self.cleanData(self.codes_tmp)
        self.len = len(self.codes)  # 记录样本的长度
        self.labels = reader.loc[:, 'label'].values  # 标签列表
        #self.label_list = sorted(reader.loc[:, 'label'].unique())  # 去重排序后的标签列表
        self.label_dict = self.getLabelDict()  
        self.label_num = len(self.train_label_list)  # 不同标签的个数  最终分类的总类别数
        if is_train_set:
            self.word2index=self.codeDic()
            self.dicnum=len(self.word2index)

    # 根据索引获取code和对应的label
    def __getitem__(self, index):
        return self.codes[index], self.label_dict[self.labels[index]]

    # 返回数据集的长度
    def __len__(self):
        return self.len

    # 根据索引获取标签值
    def idx2label(self, index):
        return self.label_list[index]

    # 获取标签种类数
    def getLabelNum(self):
        return self.label_num

    # 获得标签对应的分类序号
    def getLabelDict(self):
        label_dict = dict()
        for idx, Label in enumerate(self.train_label_list, 0):
            label_dict[Label] = idx
        return label_dict
    
    #生成词和下标数字对应的字典 {'int':0,'return':1,……}
    def codeDic(self):
        ss=set()
        for s in self.codes:
            ss.update(set(s))
        #转化为列表并排序使得符号和索引一一对应
        ss=list(ss)
        ss.sort()
        #词->索引
        #index=0位置空出来，0对应在字典中找不到的字符串
        word2index={}
        #词表中未出现的单词
        word2index["<unk>"]=0
        #句子添加的padding
        word2index["<pad>"]=1
        j=2
        for word in ss:
            word2index[word]=j
            j+=1      
        return word2index
      
    #将'\n' '\t'以及无用空格全部删去
    #返回分词列表列表
    def cleanData(self,code1):
        r=''
        i=0
        codes=[]
        while i<len(code1):
            s=code1[i].split("\n")
            r=''
            for ss in s:
                r+=ss.strip()
            sentences=word_tokenize(r)
            interpunctuations = [',', '.', ':', ';', '(', ')', '[', ']','{','}','main']
            cutwords=[word for word in sentences if word not in interpunctuations]
            codes.append(cutwords)
            i+=1
        return codes

    #得到代码分词后的最大长度
    def getMaxlen(self,code1):
        maxlen=0
        for list in code1:
            if len(list)>maxlen:
                maxlen=len(list)
        return maxlen

     #判断是否放到显卡上
    def create_tensor(self,tt):
        if USE_GPU:
            device=torch.device('cuda:0')
            tt=tt.to(device)
        return tt 