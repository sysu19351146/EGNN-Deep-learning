基于EGNN的新冠肺炎辅助诊断模型
=

任务描述
---
用构建的新冠肺炎病人肺部CT图片数据集训练图卷积神经网络，使其能够在接收到一张全新的肺部CT图片时能够判断病人是否患有新冠肺炎，


结构框架
---
![Image text](https://raw.githubusercontent.com/sysu19351146/EGNN-Deep-learning/main/img_for_readme/%E5%9B%BE%E7%89%871.png)

![Image text](https://raw.githubusercontent.com/sysu19351146/EGNN-Deep-learning/main/img_for_readme/%E5%9B%BE%E7%89%872.png)




文件作用以及运行过程
---

##YES和NO文件夹分别存放的是新冠肺炎和正常的CT数据集

##support文件夹存放的是用来做support的数据集

##pth1中放的是未使用中心裁剪训练的模型权重文件

##pth2中放的是使用了中心裁剪训练的模型权重文件

##fem.pth是预训练的权重

##data_loader.py 是数据集读取文件

##EGNN.py是主文件，运行时只需更改process_type变量名的值即可切换两种数据增强的测试结果


