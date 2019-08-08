# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch as t
from torch.autograd import Variable
import numpy as np
import cv2 as cv


class Arbitrary_ShoeprintDataset(data.Dataset):

    def __init__(self, root, fold=0, pattern=True, formats=True, types=False, thickness=6):

        # 参数注释:
        # root: 数据集根目录
        # fold: 交叉验证批次,总共5[0-4]
        # pattern: 训练集/测试集 True为训练集
        # formats: 是否为10-80岁区间数据 True为10-80区间内数据
        # types: 灰度/RGB原图 True为灰度图

        self.size = [384, 768]     # 定义resize大小，可以改成变量
        self.Pattern = pattern  # 见上方注释
        self.images_root = root  # 读取数据集根地址
        self.formats = formats  # 见上方注释
        self.fold = fold    # 见上方注释
        self.types = types  # 见上方注释
        self.thickness = thickness
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

    def div_list(self, ls, n):
        if not isinstance(ls, list) or not isinstance(n, int):
            return []
        ls_len = len(ls)
        if n <= 0 or 0 == ls_len:
            return []
        if n > ls_len:
            return []
        elif n == ls_len:
            return [[i] for i in ls]
        else:
            j = ls_len / n
            k = ls_len % n
            ### j,j,j,...(前面有n-1个j),j+k
            # 步长j,次数n-1
            ls_return = []
            for i in range(0, int((n - 1) * j + 1), int(j)):
                ls_return.append(ls[i:int(i + j)])
            # 算上末尾的j+k
            ls_return.append(ls[int((n - 1) * j):])
            # for i in ls_return:
            #     print(len(ls_return))

            return ls_return
    def _read_txt_file(self):

        self.data_path = []   # 创建data地址数组
        self.label_path = []    # 创建label数组

        for i in range(len(os.listdir(self.images_root+'Label/'))):
            # print(self.images_root+'Label/'+os.listdir(self.images_root+'Label/')[i])
            read_txt = open(self.images_root+'Label/'+os.listdir(self.images_root+'Label/')[i], 'r').readlines()
            for j in read_txt:
                if self.types:
                    self.data_path.append(self.images_root+'Datas/' + j.split(' ')[0])
                else:
                    self.data_path.append(self.images_root + 'Data/' + j.split(' ')[0])
                # 这里对标签进行了修改，进行11年龄段预测,如果要预测全年龄请切换一下
                self.label_path.append(str(i))
                # 全年龄预测
                # self.label_path.append(j.split(' ')[1][:-1])
                # 部分区间，请在下方写一个过滤

                # print(self.images_root+'Data/' + j.split(' ')[1][:-1])
        self.data_path_cut = self.div_list(self.data_path, 5)
        self.label_path_cut = self.div_list(self.label_path, 5)
        self.data_path = []  # 创建data地址数组
        self.label_path = []  # 创建label数组

        if self.Pattern:
            for k in range(5):
                if k != self.fold:
                    for x in self.data_path_cut[k]:
                        self.data_path.append(x)
                    for x in self.label_path_cut[k]:
                        self.label_path.append(x)
        else:
            for k in range(5):
                if k == self.fold:
                    for x in self.data_path_cut[k]:
                        self.data_path.append(x)
                    for x in self.label_path_cut[k]:
                        self.label_path.append(x)

    def __getitem__(self, index):   # 这个函数负责根据所以去读取数据，index是程序调用的，当然你也可以自己去调用看看写的对不对
        '''
        return the data of one image
        '''
        data_path = self.data_path[index]			# 根据出入的index去读取第index-1个图像的地址
        label = self.label_path[index]				# 根据出入的index去读取第index-1个数据label

        if self.types:
            datas = t.zeros(self.thickness, self.size[0], self.size[1])
        else:
            datas = t.zeros(self.thickness * 3, self.size[0], self.size[1])
        channel = 0
        for i in range(self.thickness):
            data = Image.open(data_path + os.listdir(data_path)[i % len(os.listdir(data_path))])
            data = self.transforms_data(data)
            if self.types:
                datas[channel] = data
                channel += 1
            else:
                datas[channel:channel+3] = data
                channel += 3

        label = np.array(int(label))				# 由于我们读取的label类型是str所以要转化
        label = t.from_numpy(label)
        label = label.long()
        # data = Image.open(self.images_root+data_path)				# data的读取就很简单了，用PIL打开
        # data = self.transforms_data(data)			# 在调用之前写好的转化函数就好了，ToTensor函数会自动把它变成torch格式，这个可以直接去打印一下

        return datas, label 							# 后面就返回就行了，返回data和label

    def __len__(self):
        # return len(self.data_path)                  # 这个函数是告诉程序数据集究竟有多大，可以直接写个具体的值，不过写超了会报错，out of range!
        return 1000

