import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        # self.zs = self.get_zs(self.zs_size)
        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))


        #
        # self.block1_1 = ResNetBlock(16, 16)
        # self.block1_2 = ResNetBlock(16, 16)
        # self.block1_3 = ResNetBlock(16, 16)
        # self.block1_4 = ResNetBlock(16, 16)
        # self.block1_5 = ResNetBlock(16, 16)
        # self.block1_6 = ResNetBlock(16, 16)
        #
        # self.block2_1 = ResNetBlock(16, 32, True)
        # self.block2_2 = ResNetBlock(32, 32)
        # self.block2_3 = ResNetBlock(32, 32)
        # self.block2_4 = ResNetBlock(32, 32)
        # self.block2_5 = ResNetBlock(32, 32)
        # self.block2_6 = ResNetBlock(32, 32)
        #
        # self.block3_1 = ResNetBlock(32, 64, True)
        # self.block3_2 = ResNetBlock(64, 64)
        # self.block3_3 = ResNetBlock(64, 64)
        # self.block3_4 = ResNetBlock(64, 64)
        # self.block3_5 = ResNetBlock(64, 64)
        # self.block3_6 = ResNetBlock(64, 64)

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64,10)

    # def get_zs(self, zs_size):
    #
    #     zs = []
    #     for z in zs_size:
    #         w = []
    #         a, b = z
    #         for i in range(a):
    #             q = []
    #             for j in range(b):
    #                 q.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))
    #             w.append(q)
    #         zs.append(w)
    #
    #     return zs
    #
    # def gen_weights(self, zs):
    #     ww = []
    #     for h in zs:
    #         w = []
    #         for k in h:
    #             w.append(self.hope(k))
    #         ww.append(torch.cat(w,dim=1))
    #     return torch.cat(ww,dim=0)

    def forward(self, x):
        count = 0

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)

        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count+1])
        # x = self.block1_1(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block1_2(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block1_3(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block1_4(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block1_5(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block1_6(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_1(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_2(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_3(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_4(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_5(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block2_6(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_1(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_2(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_3(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_4(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_5(x, w1, w2)
        # count += 2
        #
        # w1 = self.gen_weights(self.zs[count])
        # w2 = self.gen_weights(self.zs[count + 1])
        # x = self.block3_6(x, w1, w2)
        # count += 2

        x = self.global_avg(x)
        x = self.final(x.view(-1,64))

        return x
