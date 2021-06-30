# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 11:00:08 2021

@author: 53412




import torch
import torch.nn as nn
from data_loader import data_load
from tqdm import tqdm
import math
import os



class Pretrain(nn.Module):
    """
    预训练模块
    """
    def __init__(self,in_chanel,out_chanel):
        super(Pretrain,self).__init__()
        self.fc1=nn.Linear(in_chanel,out_chanel)
        self.act1 = nn.Sigmoid()

    def forward(self,x):
        y1=self.fc1(x)
        y1=self.act1(y1)
        return y1


class Bottleneck(nn.Module):
    """
    embedding层的卷积模块
    """
    def __init__(self,in_chanel,out_chanel):
        super(Bottleneck,self).__init__()
        self.out_chanel=out_chanel
        self.in_ch=in_chanel
        self.conv1=nn.Conv2d(in_chanel,out_chanel,kernel_size=3,stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_chanel)
        self.max1=nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.de=nn.Conv2d(in_chanel,out_chanel,1)
    def forward(self,x):
        y1=self.conv1(x)
        y1=self.bn1(y1)
        y1=y1+self.de(x)
        y1=self.relu1(y1)
        y1=self.max1(y1)
        return y1
    
class Fembed(nn.Module):
    """
    特征嵌入的模块
    """
    def __init__(self,in_ch,blocks,out):
        super(Fembed,self).__init__()
        self.in_ch=in_ch
        self.blocks_=blocks
        self.num=len(blocks)
        self.layers=[]
        for i in range(self.num):
            if i!=0:
                self.layers.append(Bottleneck(blocks[i-1],blocks[i]))
            else:
                self.layers.append(Bottleneck(in_ch,blocks[0]))
        self.layers=nn.Sequential(*self.layers)
        self.fc=nn.Linear(6400,out)
        self.bn1 = nn.BatchNorm1d(out)
        self.relu1 = nn.LeakyReLU(0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        y=self.layers(x)
        y=y.view(y.shape[0],-1)
        y=self.fc(y)
        y=self.bn1(y)
        y=self.relu1(y)
        return y
  
class Bottleneck2(nn.Module):
    """
    更新边网络的卷积模块
    """
    def __init__(self,in_chanel,out_chanel):
        super(Bottleneck2,self).__init__()
        self.out_chanel=out_chanel
        self.in_ch=in_chanel
        self.conv1=nn.Conv2d(in_chanel,out_chanel,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_chanel)
        self.relu1 = nn.ReLU6(inplace=True)
    def forward(self,x):
        y1=self.conv1(x)
        y1=self.bn1(y1)
        y1=self.relu1(y1)
        return y1 
  
class FV(nn.Module):
    """
    更新边的网络
    """
    def __init__(self,in_ch,blocks,out):
        super(FV,self).__init__()
        self.in_ch=in_ch
        self.blocks_=blocks
        self.num=len(blocks)
        self.layers=[]
        for i in range(self.num):
            if i!=0:
                self.layers.append(Bottleneck2(blocks[i-1],blocks[i]))
            else:
                self.layers.append(Bottleneck2(in_ch,blocks[0]))
        self.layers=nn.Sequential(*self.layers)
        self.fc=nn.Linear(128,out)
        self.act1 = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        y=self.layers(x.view(x.shape[0],128,1,1))
        y=y.view(y.shape[0],-1)
        y=self.fc(y)
        y=self.act1(y)
        return y

class MLP(nn.Module):
    """
    计算结点的全连接网络
    """
    def __init__(self,in_chanel,h,out_chanel):
        super(MLP,self).__init__()
        self.out_chanel=out_chanel
        self.in_ch=in_chanel
        self.h=h
        self.fc1=nn.Conv2d(in_chanel,h,kernel_size=1,bias=False)
        self.fc2=nn.Conv2d(h,out_chanel,kernel_size=1,bias=False)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
    def forward(self,x):
        y1=self.fc1(x.view(x.shape[0]*11,256,1,1))
        y1=self.act1(y1)
        y1=self.fc2(y1)
        y1=self.act2(y1)
        return y1.view(x.shape[0],11,128) 



def count_new_node(nodes,edges,mlp_net):
    """
    更新结点
    """
    new_nodes=torch.randn(nodes.shape[0],nodes.shape[1],256).cuda()
    new_no=torch.randn(nodes.shape[0],nodes.shape[1],128).cuda()
    node_all=nodes.shape[1]
    old_all_1=torch.zeros(batch_size).cuda()
    old_all_2=torch.zeros(batch_size).cuda()
    for k in range(nodes.shape[0]):
        old_all_1=torch.sum(torch.sum(edges[:,:,:,0],1),1)
        old_all_2=torch.sum(torch.sum(edges[:,:,:,1],1),1)
        for i in range(node_all):
            for j in range(node_all):
                new_nodes[k][i][:128]=new_nodes[k][i][:128]+edges[k][i][j][0]*nodes[k][j]/old_all_1[k]
                new_nodes[k][i][128:]=new_nodes[k][i][128:]+edges[k][i][j][1]*nodes[k][j]/old_all_1[k]
    new_no=mlp_net(new_nodes)
        
    return new_no

def count_new_edge(nodes,edges,fv_net):
    """
    更新边
    """
    node_all=nodes.shape[1]
    new_edges=torch.Tensor(edges.shape).cuda()
    old_all_1=torch.zeros(nodes.shape[0]).cuda()
    new_all_1=torch.zeros(nodes.shape[0]).cuda()
    old_all_2=torch.zeros(nodes.shape[0]).cuda()
    new_all_2=torch.zeros(nodes.shape[0]).cuda()
    
    for i in range(node_all):
        for j in range(node_all):
            new_edges[:,i,j,0]=(edges[:,i,j,0:1].expand(nodes.shape[0],1)*fv_net(torch.abs(nodes[:,i]-nodes[:,j]))).squeeze()
            new_edges[:,i,j,1]=(edges[:,i,j,1:].expand(nodes.shape[0],1)*(1-fv_net(torch.abs(nodes[:,i]-nodes[:,j])))).squeeze()
    old_all_1=torch.sum(torch.sum(edges[:,:,:,0],1),1).detach()
    new_all_1=torch.sum(torch.sum(new_edges[:,:,:,0],1),1).detach()
    old_all_2=torch.sum(torch.sum(edges[:,:,:,1],1),1).detach()
    new_all_2=torch.sum(torch.sum(new_edges[:,:,:,1],1),1).detach()
    division_1=new_all_1/old_all_1
    division_2=new_all_2/old_all_2
    for k in range(nodes.shape[0]):
        new_edges[k,:,:,0]=new_edges[k,:,:,0]/division_1[k]
        new_edges[k,:,:,1]=new_edges[k,:,:,1]/division_2[k]
    return new_edges

def edge_init(edges,label):
    """
    边的初始化
    """
    edges_num=edges.shape[1]
    for i in range(edges_num-1):
        for j in range(edges_num-1): 
            if label[i]==label[j]:
                edges[:,i,j,0]=1
                edges[:,i,j,1]=0
            else:
                edges[:,i,j,0]=0
                edges[:,i,j,1]=1
    edges[:,10,:,:]=0.5
    edges[:,:,10,:]=0.5
    return edges


def count_possible(edges,nodes,label):
    """
    计算属于两个类别的概率
    """
    possible=torch.randn((nodes.shape[0],2)).cuda()
    mid1=torch.sum(nodes[:,:5,:],1)/5
    mid2=torch.sum(nodes[:,5:10],1)/5
    dis1=torch.sum(torch.pow(mid1-nodes[:,10,:],2),1)
    dis2=torch.sum(torch.pow(mid2-nodes[:,10,:],2),1)
    sum_=dis1+dis2
    dis1=1-dis1/sum_
    dis2=1-dis2/sum_
    possible[:,0]=torch.sum(edges[:,10,:10,0]*label,1)
    possible[:,1]=torch.sum(edges[:,10,:10,0]*(1-label),1)
    sum_2=possible[:,0].detach()+possible[:,1].detach()
    possible[:,0]=possible[:,0]/sum_2+dis1
    possible[:,1]=possible[:,1]/sum_2+dis2
    return possible


def count_loss(possible,label):
    """
    计算损失函数
    """
    cross_loss= nn.CrossEntropyLoss().cuda()
    loss=0
    for i in range(len(possible)):
        loss=loss+cross_loss(possible[i],label)
    return loss


def train(epoch,data,supprt,fem_net,mlp_net,fv_net,optimizer1,optimizer2,optimizer3,L):
    """
    训练过程
    """
    for i in range(epoch):
        fem_net.train()
        mlp_net.train()
        fv_net.train()
        train_iter=1
        valid=int(len(data)*0.8)
        loss_train=0
        loss_valid=0
        with tqdm(total=len(data),desc="epoch:{}/{}".format(i+1,epoch)) as pbar:
            for data_ in data:
                if train_iter<=valid:
                    inputs,label=data_
                    inputs,label=inputs.float().cuda(),label.cuda()
                    output=fem_net(inputs)
                    for supprt_data in supprt:
                        input_q,label_q=supprt_data
                        input_q,label_q=input_q.float().cuda(),label_q.cuda()
                        output_q=fem_net(input_q)
                    all_nodes=[]
                    all_edges=[]
                    all_possible=[]
                    nodes=torch.zeros((label.shape[0],11,128)).cuda()
                    nodes[:,:10,:]=output_q
                    nodes[:,10,:]=output
                    edges=torch.zeros((label.shape[0],11,11,2)).cuda()
                    edges=edge_init(edges,label_q)
                    all_nodes.append(nodes)
                    all_edges.append(edges)
                    all_possible.append(count_possible(edges,nodes,label_q))
                    for index in range(L):
                        new_nodes=count_new_node(all_nodes[index],all_edges[index],mlp_net)
                        all_nodes.append(new_nodes)
                        new_edges=count_new_edge(nodes,edges,fv_net)
                        all_edges.append(new_edges)
                        all_possible.append(count_possible(new_edges,new_nodes,label_q))
                    loss=count_loss(all_possible,label)
                    loss_train+=loss.item()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    optimizer3.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer1.step()
                    optimizer2.step()
                    optimizer3.step()
                    pbar.set_postfix({"loss":loss_train/train_iter})
                    pbar.update(1)
                    train_iter+=1
                else:
                    inputs,label=data_
                    inputs,label=inputs.float().cuda(),label.cuda()
                    output=fem_net(inputs)
                    for supprt_data in supprt:
                        input_q,label_q=supprt_data
                        input_q,label_q=input_q.float().cuda(),label_q.cuda()
                        output_q=fem_net(input_q)
                    all_nodes=[]
                    all_edges=[]
                    all_possible=[]
                    nodes=torch.zeros((label.shape[0],11,128)).cuda()
                    nodes[:,:10,:]=output_q
                    nodes[:,10,:]=output
                    edges=torch.zeros((label.shape[0],11,11,2)).cuda()
                    edges=edge_init(edges,label_q)
                    all_nodes.append(nodes)
                    all_edges.append(edges)
                    all_possible.append(count_possible(edges,nodes,label_q))
                    for index in range(L):
                        new_nodes=count_new_node(all_nodes[index],all_edges[index],mlp_net)
                        all_nodes.append(new_nodes)
                        new_edges=count_new_edge(nodes,edges,fv_net)
                        all_edges.append(new_edges)
                        all_possible.append(count_possible(new_edges,new_nodes,label_q))
                    loss=count_loss(all_possible,label)
                    loss_valid+=loss.item()
                    pbar.set_postfix({"val_loss":loss_valid/(train_iter-valid)})
                    pbar.update(1)
                    train_iter+=1   
        torch.save(fem_net,"pth{}/fem_epoch{}.pth".format(process_type,i+1))
        torch.save(mlp_net,"pth{}/mlp_epoch{}.pth".format(process_type,i+1))
        torch.save(fv_net,"pth{}/fv_epoch{}.pth".format(process_type,i+1))

 
def test(data,supprt,fem_net,mlp_net,fv_net,L):
    """
    测试过程
    """
    fem_net.eval()
    mlp_net.eval()
    fv_net.eval()
    acc_=0.0
    sum_=0.0
    test_iter=1
    with tqdm(total=len(data),desc="test") as pbar:
        for data_ in data:
            if test_iter==len(data):
                pbar.update(1)
                break
            inputs,label=data_
            inputs,label=inputs.float().cuda(),label.cuda()
            output=fem_net(inputs)
            for supprt_data in supprt:
                input_q,label_q=supprt_data
                input_q,label_q=input_q.float().cuda(),label_q.cuda()
                output_q=fem_net(input_q)
            all_nodes=[]
            all_edges=[]
            all_possible=[]
            nodes=torch.zeros((label.shape[0],11,128)).cuda()
            nodes[:,:10,:]=output_q
            nodes[:,10,:]=output
            edges=torch.zeros((label.shape[0],11,11,2)).cuda()
            edges=edge_init(edges,label_q)
            all_nodes.append(nodes)
            all_edges.append(edges)
            all_possible.append(count_possible(edges,nodes,label_q))
            for index in range(L):
                new_nodes=count_new_node(all_nodes[index],all_edges[index],mlp_net)
                all_nodes.append(new_nodes)
                new_edges=count_new_edge(nodes,edges,fv_net)
                all_edges.append(new_edges)
                all_possible.append(count_possible(new_edges,new_nodes,label_q))
            _,pred=torch.max(all_possible[-1],axis=1)
            acc_+=torch.sum(pred==label).item()
            sum_+=label.shape[0]
            pbar.set_postfix({"accu":acc_/sum_})
            pbar.update(1)
            test_iter+=1
               
            


def pretrain(epoch,data,fem_net,pre_train,optimizer1,optimizer2):
    """
    预训练函数
    """
    for i in range(epoch):

        train_iter=1
        valid=int(len(data)*0.9)
        loss_train=0
        loss_valid=0
        bce_loss = nn.BCELoss().cuda()
        with tqdm(total=len(data),desc="epoch:{}/{}".format(i+1,epoch)) as pbar:
            for data_ in data:
                if train_iter<=valid:
                    inputs,label=data_
                    inputs,label=inputs.float().cuda(),label.cuda()
                    output=fem_net(inputs)
                    output=pre_train(output)
                    loss=bce_loss(output,label.view(label.shape[0],1).float())
                    loss_train+=loss.item()
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer1.step()
                    optimizer2.step()
                    pbar.set_postfix({"loss":loss_train/train_iter})
                    pbar.update(1)
                    train_iter+=1
                else:
                    inputs,label=data_
                    inputs,label=inputs.float().cuda(),label.cuda()
                    output=fem_net(inputs)
                    output=pre_train(output)
                    loss=bce_loss(output,label.view(label.shape[0],1).float())
                    loss_valid+=loss.item()
                    pbar.set_postfix({"val_loss":loss_valid/(train_iter-valid)})
                    pbar.update(1)
                    train_iter+=1   
        torch.save(fem_net,"pth1/fem_pre_epoch{}.pth".format(i+1))
       

def test_pre(data,fem_net,pre_train):
    """
    预训练的测试函数
    """
    fem_net.eval()
    pre_train.eval()
    acc_=0.0
    sum_=0.0
    test_iter=1
    with tqdm(total=len(data),desc="test" )as pbar:
        for data_ in data:
            inputs,label=data_
            inputs,label=inputs.float().cuda(),label.cuda()
            output=fem_net(inputs)
            output=pre_train(output)
            pred=output>0.5
            acc_+=torch.sum(pred==label.view(label.shape[0],1)).item()
            sum_+=label.shape[0]
            pbar.set_postfix({"accu":acc_/sum_})
            pbar.update(1)
            test_iter+=1








seed = 224        #设置随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


process_type=2           #1表示未使用中心裁剪，2代表使用中心裁剪
batch_size=8             #batch大小
epoch=0                  #训练轮数
L=3                      #迭代轮数
train_data,test_data,supprt_data=data_load(process_type)     #载入数据


#初始化网络
fem_net=Fembed(3,[64,96,128,256],128)
fv_net=FV(128,[256,256,128,128],1)
mlp_net=MLP(256,128,128)
pre_train=Pretrain(128,1)

#构建优化器
optimizer1 = torch.optim.Adam(fem_net.parameters(),lr=1e-4,weight_decay=1e-6)
optimizer2 = torch.optim.Adam(mlp_net.parameters(),lr=1e-3,weight_decay=1e-6)
optimizer3 = torch.optim.Adam(fv_net.parameters(),lr=1e-3,weight_decay=1e-6)
optimizer4 = torch.optim.Adam(pre_train.parameters(),lr=1e-3,weight_decay=1e-6)

#转GPU训练
fem_net=fem_net.cuda()
fv_net=fv_net.cuda()
mlp_net=mlp_net.cuda()
pre_train=pre_train.cuda()


#fem_net=torch.load("fem.pth")       #加载预训练的权重
#加载权重文件
pth_path="pth{}/mlp.pth".format(process_type)
if os.path.exists(pth_path):
    mlp_net=torch.load(pth_path)
    fem_net=torch.load("pth{}/fem.pth".format(process_type))
    fv_net=torch.load("pth{}/fv.pth".format(process_type))
    print("weight_load")


#train(epoch,train_data,supprt_data,fem_net,mlp_net,fv_net,optimizer1,optimizer2,optimizer3,L)   #训练
test(test_data,supprt_data,fem_net,mlp_net,fv_net,L)                                            #测试

