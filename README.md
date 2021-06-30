基于改进后的EGNN的新冠肺炎辅助诊断模型
===

任务描述
---
用构建的新冠肺炎病人肺部CT图片数据集训练修改后的EGNN模型，使其能够在接收到一张全新的肺部CT图片时能够判断病人是否患有新冠肺炎

EGNN结构框架
---
![Image text](https://raw.githubusercontent.com/sysu19351146/EGNN-Deep-learning/main/img_for_readme/%E5%9B%BE%E7%89%871.png)

![Image text](https://raw.githubusercontent.com/sysu19351146/EGNN-Deep-learning/main/img_for_readme/%E5%9B%BE%E7%89%872.png)

模型改进
---
* 改换结构：edge只在第一次时进行更新，node多次更新。
* 加入残差：在开始的embeding层换为加残差的卷积结构。
* 预训练：对开始的embedding层先进行预训练。
* 双重距离限制：在计算possible的基础上加入了原型网络的相似度限制。
![Image text](https://raw.githubusercontent.com/sysu19351146/EGNN-Deep-learning/main/img_for_readme/%E5%9B%BE%E7%89%873.png)

pytorch版本
---
* PyTorch 1.8.1

文件作用以及运行过程
---
* YES和NO文件夹分别存放的是新冠肺炎和正常的CT数据集

* support文件夹存放的是用来做support的数据集

* pth1中放的是未使用中心裁剪训练的模型权重文件

* pth2中放的是使用了中心裁剪训练的模型权重文件

* img_for_readme文件夹保存的是用来写readme的图片

* fem.pth是预训练的权重

* data_loader.py 是数据集读取文件

* EGNN.py是主文件，运行时只需更改process_type变量名的值即可切换两种数据增强的测试结果

 

参考文献
---
* Edge-Labeling Graph Neural Network for Few-shot Learning.  CVPR, 2019
* Augmented Multi-center Graph Convolutional Network for COVID-19 Diagnosis. IEEE Transactions on Industrial Informatics,2021
* BSNet: Bi-Similarity Network for Few-shot Fine-grained Image Classification. IEEE TRANSACTIONS ON IMAGE PROCESSING,2021


