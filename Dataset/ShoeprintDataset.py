# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch as t
from torch.autograd import Variable
import numpy as np
import cv2 as cv


class ShoeprintDataset(data.Dataset):

    def __init__(self, root, fold=0, pattern=True, formats=True, types=True):

        # 参数注释:
        # root: 数据集根目录
        # fold: 交叉验证批次,总共5[0-4]
        # pattern: 训练集/测试集 True为训练集
        # formats: 是否为10-80岁区间数据 True为10-80区间内数据
        # types: 灰度/RGB原图 True为灰度图

        self.size = [768, 384]     # 定义resize大小，可以改成变量
        self.Pattern = pattern  # 见上方注释
        self.images_root = root  # 读取数据集根地址
        self.formats = formats  # 见上方注释
        self.fold = fold    # 见上方注释
        self.types = types  # 见上方注释
        if self.types:  # 判断是灰度图还是RGB图
            self.transforms_data = T.Compose([
                T.Resize(self.size),    # 裁剪
                # T.CenterCrop(self.size),    # 中心裁剪
                T.ToTensor(),   # ToTensor可以把数值压缩到【0,1】之间
                T.Normalize([0.15311628], [0.30212043])   # 标准化操作，减均值除标准差
                # T.Normalize([.5，.5，.5], [.5,.5,.5])
            ])
        else:
            self.transforms_data = T.Compose([
                T.Resize(self.size),    # 裁剪
                # T.CenterCrop(self.size),    # 中心裁剪
                T.ToTensor(),   # ToTensor可以把数值压缩到【0,1】之间
                T.Normalize([0.15303094, 0.15053251, 0.06985552], [0.30209424, 0.29732634, 0.15284354])
                # 标准化操作，减均值除标准差
            ])
        self._read_txt_file()   # 读取txt索引文件

    def _read_txt_file(self):

        self.data_path = []   # 创建data地址数组
        self.label_path = []    # 创建label数组
        # label_txt是索引文件的地址，根据条件去赋值
        if self.types:  # 判断是否为灰度数据集
            self.label_txt = self.images_root + 'Label_gray/'
        else:
            self.label_txt = self.images_root + 'Label_origin/'

        if self.Pattern:    # 判断是否为训练集
            if self.formats:    # 判断是否为10-80区间数据
                self.label_txt = self.label_txt + 'train_filter_' + str(self.fold) + '.txt'
            else:
                self.label_txt = self.label_txt + 'train_origin_' + str(self.fold) + '.txt'

        else:
            if self.formats:
                self.label_txt = self.label_txt + 'val_filter_' + str(self.fold) + '.txt'
            else:
                self.label_txt = self.label_txt + 'val_origin_' + str(self.fold) + '.txt'
        with open(self.label_txt, 'r') as f:				# 打开txt，把里面的地址一个一个读进去
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')		# 吃掉分割符号
                self.data_path.append(item[0])		# 这个是数据地址
                self.label_path.append(item[1])		# 这个是标签
        # ata_path里面就是存的图像数据的地址，label_path就是那个图像所对应的label值

    def __getitem__(self, index):   # 这个函数负责根据所以去读取数据，index是程序调用的，当然你也可以自己去调用看看写的对不对
        '''
        return the data of one image
        '''
        data_path = self.data_path[index]			# 根据出入的index去读取第index-1个图像的地址
        label = self.label_path[index]				# 根据出入的index去读取第index-1个数据label
        label = np.array(int(label))				# 由于我们读取的label类型是str所以要转化
        label = t.from_numpy(label)					
        label = label.long()						
        data = Image.open(self.images_root+data_path)				# data的读取就很简单了，用PIL打开
        data = self.transforms_data(data)			# 在调用之前写好的转化函数就好了，ToTensor函数会自动把它变成torch格式，这个可以直接去打印一下

        return data, label 							# 后面就返回就行了，返回data和label

    def __len__(self):
        return len(self.data_path)                  # 这个函数是告诉程序数据集究竟有多大，可以直接写个具体的值，不过写超了会报错，out of range!
        # return 1000
