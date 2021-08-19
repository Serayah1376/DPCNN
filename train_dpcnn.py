# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:08:25 2021

@author: 10983
"""

import torch
import time
import math

class Train_Valid():
    def __init__(self,model,trainloader,validloader,testloader,start,trainset,validset,testset):
        self.model=model
        self.trainloader=trainloader
        self.validloader=validloader
        self.testloader=testloader
        self.start=start
        self.trainset=trainset
        self.validset=validset
        self.testset=testset
        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        

    def train(self):
        valid_acc_list=[]
        test_acc_list=[]
        #循环100轮
        loss_list=[]
        #min_loss=100  #初始化一个最小损失值
        best_acc=0
        for epoch in range(1,31): 
            total_loss=0
            for i,(inputs,target) in enumerate(self.trainloader,1):
                self.optimizer.zero_grad()
                output=self.model.forward(inputs)
                loss=self.criterion(output,target)
                loss.backward()
                self.optimizer.step()
                #self.scheduler.step()
            
                total_loss+=loss.item()
                if i%100==0:
                    print(f'[{self.time_since(self.start)}] Epoch {epoch}',end='')
                    print(f'[{i * len(inputs)}/{len(self.trainset)}]',end='')
                    print(f'loss={total_loss/(i*len(inputs))}')
               
            '''
            #如果出现损失值的最新低值，则保存网络模型
            if (total_loss/len(trainset))<min_loss:
                print('**********************')
                min_loss=total_loss/len(trainset) #更新值
                torch.save(classifier.state_dict(),'C:/Users/10983/py入门/GRUClassifier/net_params.pkl')
            '''
            loss_list.append(total_loss/len(self.trainset))
            #训练一轮的结束进行一次验证，获取准确率
            valid_acc=self.valid(epoch)
            #学习率优化器
            #self.scheduler.step(valid_acc)
            test_acc=self.test(epoch)
            self.scheduler.step(test_acc)
            
            #保存效果最好的模型
            if test_acc>best_acc:
                best_acc=test_acc
                torch.save(self.model, 'C:/Users/10983/py入门/GRUClassifier/best_model')
                
            valid_acc_list.append(valid_acc)
            test_acc_list.append(test_acc)
            
        return valid_acc_list,test_acc_list,loss_list
    
    def valid(self,epoch):
        correct,valid_total_loss=0,0
        total=len(self.validset)
        print()
        print('Evaluating trained model(use valid data)...')

        with torch.no_grad():
            for i,(inputs,target) in enumerate(self.validloader,1):
                output=self.model.forward(inputs)
                valid_loss=self.criterion(output,target)
                pred=output.max(dim=1,keepdim=True)[1]
                correct+=pred.eq(target.view_as(pred)).sum().item()
                
                valid_total_loss+=valid_loss.item()
                if i%20==0:
                    print(f'[{self.time_since(self.start)}] Epoch {epoch}',end='')
                    print(f'[{i * len(inputs)}/{len(self.validset)}]',end='')
                    print(f'valid_loss={valid_total_loss/(i*len(inputs))}')
               
        percent='%.2f' % (100 * correct /total)
        print(f'Valid set: Accuracy {correct}/{total} {percent} %')
            
        return correct / total
    
    def test(self,epoch):
        correct,test_total_loss=0,0
        total=len(self.testset)
        print()
        print('Evaluating trained model(use test data)...')
        
        with torch.no_grad():
            for i,(inputs,target) in enumerate(self.testloader,1):
                output=self.model.forward(inputs)
                test_loss=self.criterion(output,target)
                pred=output.max(dim=1,keepdim=True)[1]
                correct+=pred.eq(target.view_as(pred)).sum().item()
                
                test_total_loss+=test_loss.item()
                if i%20==0:
                    print(f'[{self.time_since(self.start)}] Epoch {epoch}',end='')
                    print(f'[{i * len(inputs)}/{len(self.testset)}]',end='')
                    print(f'test_loss={test_total_loss/(i*len(inputs))}')
                
        percent='%.2f' % (100 * correct /total)
        print(f'Test set: Accuracy {correct}/{total} {percent} %')
        print()
        
        return correct/total
        
    
    def time_since(self,since):
        s=time.time()-since
        m=math.floor(s/60)
        s-=m*60
        return '%dm %ds'% (m,s)