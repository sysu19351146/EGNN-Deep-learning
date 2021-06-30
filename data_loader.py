# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:45:50 2021

@author: 53412
"""
import numpy as np
import shutil
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



#数据增强1
data_transform1 = transforms.Compose(
      [ transforms.ToTensor(),
        transforms.Resize((84,84)),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        transforms.RandomRotation(10),
     ])

#数据增强2
data_transform2 = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
       transforms.RandomRotation(10),
       transforms.Resize((250,150)),
       transforms.CenterCrop((200,120)),
       transforms.Resize((84,84))
     ])




#训练与测试集数据类
class my_dataset(Dataset):
    def __init__(self, store_path, splits,data_transform=None):
        self.store_path = store_path
        self.split = splits
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        for split in splits:
            for file in glob.glob(split + '/' + store_path + '/*.jpg'):
                cur_path = file.replace('\\', '/')
                if split=="YES":
                    cur_label=1
                else:
                    cur_label=0
                self.img_list.append(cur_path)
                self.label_list.append(cur_label)
 
    def __getitem__(self, item):
        img=Image.open(self.img_list[item]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label
 
    def __len__(self):
        return len(self.img_list)


#支持集数据类
class supprt_data(Dataset):
    def __init__(self, store_path, splits,data_transform=None):
        self.store_path = store_path
        self.split = splits
        self.transforms = data_transform
        self.img_list = []
        self.label_list = []
        for split in splits:
            for file in glob.glob(store_path + '/' + split + '/*.jpg'):
                cur_path = file.replace('\\', '/')
                if split=="YES":
                    cur_label=1
                else:
                    cur_label=0
                self.img_list.append(cur_path)
                self.label_list.append(cur_label)
 
    def __getitem__(self, item):
        img=Image.open(self.img_list[item]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.label_list[item]
        return img, label
 
    def __len__(self):
        return len(self.img_list)
    
def data_load(process_type):
    #数据存放位置
    train_data_path="train"
    test_data_path="test"
    supprt_path="support"
    listdir=["YES","NO"]

    #导入训练、测试、支持集的数据
    data_process="data_transform"+str(process_type)  #数据增强的类别
    train_dataset=my_dataset(train_data_path,listdir,data_transform=eval(data_process))
    train_dataset_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0) 
    test_dataset=my_dataset(test_data_path,listdir,data_transform=eval(data_process))
    test_dataset_loader=DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0) 
    supprt_dataset=supprt_data(supprt_path,listdir,data_transform=eval(data_process))
    supprt_data_loader=DataLoader(supprt_dataset, batch_size=10, num_workers=0)
    return train_dataset_loader,test_dataset_loader,supprt_data_loader
    



