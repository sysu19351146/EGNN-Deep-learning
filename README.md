#简介













YES和NO文件夹分别存放的是新冠肺炎和正常的CT数据集

support文件夹存放的是用来做support的数据集

pth1中放的是未使用中心裁剪训练的模型权重文件

pth2中放的是使用了中心裁剪训练的模型权重文件

fem.pth是预训练的权重

data_loader.py 是数据集读取文件

EGNN.py是主文件，运行时只需更改process_type变量名的值即可切换两种数据增强的测试结果


