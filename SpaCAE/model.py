import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.init as init


class GraphConvolution(Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, feature_ori, feature_aug, adjacency):
        support_ori = torch.mm(feature_ori, self.weight)
        output_ori = torch.spmm(adjacency, support_ori)
        """输入增强矩阵，输出卷积层降维结果"""
        support_aug = torch.mm(feature_aug, self.weight)
        output_aug = torch.spmm(adjacency, support_aug)

        if self.use_bias:
            output_ori += self.bias
            output_aug += self.bias
        output_ori = F.relu(output_ori)
        output_aug = F.relu(output_aug)
        return output_ori, output_aug

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GraphDeconvolution(Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphDeconvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # 初始化w

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, feature_ori, feature_aug, adjacency):
        support_ori = torch.mm(feature_ori, self.weight)
        output_ori = torch.spmm(adjacency, support_ori)
        """输入增强矩阵，输出卷积层降维结果"""
        support_aug = torch.mm(feature_aug, self.weight)
        output_aug = torch.spmm(adjacency, support_aug)

        if self.use_bias:
            output_ori += self.bias
            output_aug += self.bias
        output_ori = F.relu(output_ori)
        output_aug = F.relu(output_aug)
        return output_ori, output_aug

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class CGAE(Module):
    def __init__(self,
                 input_dim,
                 laten_dim,
                 output_dim):
        super(CGAE, self).__init__()

        self.z_layer = GraphConvolution(input_dim, laten_dim)
        self.x_hat_layer = GraphDeconvolution(laten_dim, output_dim)

    def forward(self, x_ori, x_aug, graph):
        """  encode  """
        z_ori, z_aug = self.z_layer(feature_ori=x_ori, feature_aug=x_aug, adjacency=graph)
        """  decode  """
        xhat_ori, xhat_aug = self.x_hat_layer(feature_ori=z_ori, feature_aug=z_aug, adjacency=graph)

        return z_ori, z_aug, xhat_ori, xhat_aug
