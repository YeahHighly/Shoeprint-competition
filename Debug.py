# -*- coding: UTF-8 -*-
import time
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
from torch.autograd import Variable
import torch
import torchvision.models as models

from torch.optim import SGD, Adam, lr_scheduler

from Dataset import ShoeprintDataset
from Network import se_resnet20
from Util import Criterion


def dir_exit(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    # 开始加载数据集
    print("Loading dataset...")
    data_root = '/home/dataset/ShoeprintDataset/'   # 数据集地址，一般不用改，服务器不同要改一下
    save_root = '/home/hly/Code/Shoeprint-competition/Save/'    # 模型保存地址，这个大家要改一下，写自己的路径
    dir_exit(save_root)     # 若模型存储地址不存在就创建
    # 初始化数据集，参数内容请参考Dataset内的ShoeprintDataset
    train_data = ShoeprintDataset(root=data_root, fold=0, pattern=True,  formats=True, types=True)
    val_data = ShoeprintDataset(root=data_root, fold=0, pattern=False,  formats=True, types=True)
    batch_size_train = 5   # 训练集batch_size
    batch_size_val = 5  # 验证集batch_size
    num_workers = 8     # 数据集加载线程数
    # criterion = nn.CrossEntropyLoss()       # 定义损失函数
    criterion = Criterion(cumulative=2, c=0.5, alpha=1)
    learning_rate = 1e-4    # lr值
    num_epoches = 20    # 训练批次

    # 创建训练集data-loader
    train_dataloader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    print('train dataset len: {}'.format(len(train_dataloader.dataset)))

    # 创建训练集data-loader
    val_dataloader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)
    print('val dataset len: {}'.format(len(val_dataloader.dataset)))

    # 检查输出数据格式
    # for batch_datas,batch_labels in train_dataloader:
    #     print(batch_datas.size(), batch_labels.size())
    #     print(batch_datas.type(), batch_labels.type())
    #     break

    # Network创建

    # SENet示例
    model = se_resnet20(num_classes=71, reduction=16)
    # print(model)

    # ResNet34示例
    # model = models.resnet34(pretrained=True)
    # model.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
    # model.fc = nn.Linear(512, 71)

    # ResNet101示例
    # model = models.resnet101(pretrained=True)
    # model.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
    # model.fc = nn.Linear(2048, 71)

    # 是否可使用GPU
    use_gpu = torch.cuda.is_available()

    # 多卡并行模型初始化
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0]).cuda()      # 修改这里去设置那几张卡

    # 模型和损失函数都放上cuda
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # 测试网络格式
    # print(model)
    # test_data = torch.randn((2, 3, 384, 768)).cuda()
    # print(format(test_data.size()))
    # test_out = model(test_data).cuda()
    # # test_out = torch.FloatTensor([[3.1, .4, .9, 0.4], [0, 0, 0.7, 0.3]]).cuda()
    # print(format(test_out.size()))
    # test_label = torch.LongTensor([0, 0]).cuda()
    # print(format(test_label.size()))
    # loss = criterion(test_out, test_label)
    # print(format(loss))
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # 优化器设置，可根据自己的要求去换，本分调用了自定义优化策略lr_scheduler去进行lr的衰减
    optimizer = optim.Adam(model.parameters(), learning_rate, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epoches)), 0.999)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # learning rate changed every epoch

    # 开始训练
    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        scheduler.step(epoch)
        # for param_group in optimizer.param_groups:
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
        print('*' * 10)
        epoch_loss = 0.0
        for i, data in enumerate(train_dataloader, 1):
            img, label = data
            if use_gpu:
                img = Variable(img.cuda())
                label = Variable(label.cuda())
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # 每个50个小batch显示训练集分数
            if i % 10 is 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, i,
                                                                   len(train_dataloader.dataset) / batch_size_train,
                                                                   loss.item()))

        # 若要保存模型把下面的注释去一下
        # if (epoch + 1) % 10 == 0:
        #     torch.save(model, save_root + 'Epoch-' + str(epoch + 1) + '.pth')

        # 验证集跑分，这里设置了每个epoch都测试，可以根据需要去改，%25之类的
        if (epoch + 1) % 1 == 0:
            model.eval()
            running_acc = 0.0
            for j, data in enumerate(val_dataloader, 1):
                img, label = data
                if use_gpu:
                    img = Variable(img.cuda())
                    label = Variable(label.cuda())
                else:
                    img = Variable(img)
                    label = Variable(label)
                out = model(img)
                loss = criterion(out, label)
                running_acc += loss.item()
            print("===> Epoch[{}]: AccLoss: {:.4f}".format(epoch, running_acc / (
                    len(val_dataloader.dataset) / batch_size_val)))



