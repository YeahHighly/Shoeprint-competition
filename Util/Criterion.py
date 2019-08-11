# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


class Criterion(nn.Module):
    def __init__(self, cumulative=2, c=0.5, alpha=1, lambda_list=[1.0, 0.2, 0.001]):
        # 1.0 0.2 0.001
        super(Criterion, self).__init__()
        self.Cumulative = cumulative
        self.ReLU = nn.ReLU(inplace=True)
        self.item = cumulative
        self.Epsilon = torch.FloatTensor([self.Cumulative]).cuda()
        self.C = torch.FloatTensor([c]).cuda()
        self.Alpha = torch.FloatTensor([alpha]).cuda()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        self.lambda_list = torch.Tensor(lambda_list)

    # 计算mcs_loss[cumulative]的值
    def mcs_loss(self, out, label):
        out = out.argmax(dim=1).float().requires_grad_(requires_grad=True)
        label = label.float().requires_grad_(requires_grad=True)

        mcs_loss = self.Epsilon - (torch.abs(out - label)) + 1
        mcs_loss = self.ReLU(mcs_loss).sum() / (out.size()[0] * (self.Epsilon + 1))
        return mcs_loss

    # 计算CS_nu的值
    def cs_score(self, nu, out, label):
        out = out.argmax(dim=1).float().requires_grad_(requires_grad=True)
        label = label.float().requires_grad_(requires_grad=True)
        nus = torch.FloatTensor([nu]).cuda()

        cs_score = nus - (torch.abs(out - label)) + 1
        cs_score = torch.gt(self.ReLU(cs_score), 0).sum().float() / out.size()[0]
        return cs_score

    # 计算1-flb_mcs的值flb_mcs是用幂律定律对CS的值做了幂律分配，C与Alpha值见构造函数
    def one_reduce_flb_mcs_loss(self, out, label):
        flb_loss = self.cs_score(0, out, label) * (self.C * (1**(-self.Alpha)))

        for i in range(self.item):
            flb_loss += self.cs_score(i+1, out, label) * (self.C * ((i+2)**(-self.Alpha)))

        return (1-flb_loss).requires_grad_(requires_grad=True)

    # 计算交叉熵loss
    def cross_entropy_loss(self, out, label):

        return self.CrossEntropyLoss(out, label)

    # 计算MSE_loss
    def mse_loss(self, out, label):
        out = out.argmax(dim=1).float().requires_grad_(requires_grad=True)
        label = label.float().requires_grad_(requires_grad=True)
        return self.MSELoss(out, label)

    # 总loss,请在此处天马行空，例子是MCS，交叉熵，MSE的组合形势，可以自己去修改
    def forward(self, out, label):
        # loss_list = torch.Tensor([self.one_reduce_flb_mcs_loss(out, label), self.cross_entropy_loss(out, label),
        #                           self.mcs_loss(out, label)]).requires_grad_(requires_grad=True)

        # loss_list包括了上述的三种loss，与相应的lambda值进行对位相乘
        # total_loss = loss_list.mul(self.lambda_list).sum()
        total_loss = self.cross_entropy_loss(out, label)
        return total_loss


# if __name__ == '__main__':
#     criterion = Criterion(cumulative=2, c=0.5, alpha=1)
#     # # # 测试网络格式
#     # # # print(model)
#     # # # test_data = torch.randn((2, 3, 384, 768)).cuda()
#     # # # print(format(test_data.size()))
#     # # # test_out = model(test_data).cuda()
#     test_out = torch.FloatTensor([[3.1, -0.4, .9, 0.4], [0, 0, 0.7, 0.3]]).cuda()
#     # # # print(format(test_out.size()[0]))
#     test_label = torch.LongTensor([0, 0]).cuda()
#     # # # print(format(test_label.size()))
#     # # loss_0 = criterion.cs_score(0, test_out, test_label)
#     # # loss_1 = criterion.cs_score(1, test_out, test_label)
#     loss = criterion(test_out, test_label)
#     print(format(loss))



