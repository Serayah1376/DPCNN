# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:07:23 2021

@author: 10983
"""

import torch
import torch.nn.functional as F


class DPCNN(torch.nn.Module):
    def __init__(self,vocab_size,embed_dim,class_num,filter_num):
        super(DPCNN,self).__init__()
        self.class_num=class_num  #要分类的种类数
        self.filter_num=filter_num  #卷积核数目
        
        self.embedding=torch.nn.Embedding(vocab_size,embed_dim)
        
        self.conv_region=torch.nn.Conv2d(1,self.filter_num,(3,embed_dim),stride=1)
        self.conv=torch.nn.Conv2d(self.filter_num,self.filter_num,(3,1),stride=1)
        self.max_pool=torch.nn.MaxPool2d(kernel_size=(3,1),stride=2)
        self.padding1=torch.nn.ZeroPad2d((0,0,1,1))
        self.padding2=torch.nn.ZeroPad2d((0,0,0,1))
        self.relu=torch.nn.ReLU()
        self.fc=torch.nn.Linear(self.filter_num,self.class_num)
        
    def forward(self,x):
        x=self.embedding(x)
        x=x.unsqueeze(1)
        x=self.conv_region(x)
        
        x=self.padding1(x)
        x=self.relu(x)
        x=self.conv(x)
        x=self.padding1(x)
        x=self.relu(x)
        x=self.conv(x)
        while x.size()[2] >2:
            x=self._block(x)
        x=x.squeeze()
        x=self.fc(x)
        return x
    
    def _block(self,x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x
        
        

    
    
    
    
    
    
    
    
    