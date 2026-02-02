import torch
from torch import nn
import torch.nn.functional as F
from . import base

#class NeRFMLP(base.baseModel):
    #def __init__(self, **params):
        #super(NeRFMLP, self).__init__(params)
        #self.coords_MLP = nn.ModuleList(
            #[nn.Linear(self.in_ch, self.netW), *[nn.Linear(self.netW + self.in_ch, self.netW) if i in self.skips else nn.Linear(self.netW, self.netW) for i in range(self.netD - 1)]]
        #)
        #self.out_MLP = nn.Linear(self.netW, self.out_ch)

    #def forward(self, x):
        #x = self.embed(x)
        #h = x
        #for idx, mlp in enumerate(self.coords_MLP):
            #h = torch.cat([x, h], -1) if idx in self.skips else F.relu(mlp(h)) 
        #out = self.out_MLP(h)
        #return out 

class NeRFMLP(base.baseModel):
    def __init__(self, **params):
        # 初始化父类
        super(NeRFMLP, self).__init__(params)
        
        # 初始化全连接层列表
        self.coords_MLP = nn.ModuleList()
        
        # 首先添加输入层
        input_layer = nn.Linear(self.in_ch, self.netW)
        self.coords_MLP.append(input_layer)
        
        # 添加隐藏层
        for i in range(self.netD - 1):
            if i in self.skips:
                layer = nn.Linear(self.netW + self.in_ch, self.netW)
            else:
                layer = nn.Linear(self.netW, self.netW)
            self.coords_MLP.append(layer)
        
        # 输出层
        self.out_MLP = nn.Linear(self.netW, self.out_ch)

    def forward(self, x):
        # 对输入数据进行嵌入操作
        x = self.embed(x)
        
        # 初始化隐藏层的输入
        h = x
        
        # 逐层处理
        for idx, mlp in enumerate(self.coords_MLP):
            if idx in self.skips:
                # 如果当前层在跳跃连接列表中，则将原输入与当前层的输出拼接
                h = torch.cat([x, h], -1)
            else:
                # 否则，直接通过当前层，并应用ReLU激活函数
                h = F.relu(mlp(h))
        
        # 最后通过输出层
        out = self.out_MLP(h)
        
        return out


class Weight_Kernel(base.baseModel):
    def __init__(self, params, num_hidden=8, num_wide=64, num_pt=8, short_cut=False):
        super().__init__(params)
        self.short_cut = short_cut

        # Adjust the input size of the first linear layer
        input_size = self.in_ch * num_pt

        hiddens = []
        for i in range((num_hidden - 1) * 2):
            if i % 2 == 0:
                hiddens.append(nn.Linear(num_wide, num_wide))
            else:
                hiddens.append(nn.ReLU())

        self.linears = nn.Sequential(
            nn.Linear(input_size, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + self.in_ch) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, num_pt)
        )

        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)

    def forward(self, input_data):
        batch_size = input_data.shape[0]
        pts = input_data[:, :-1, :]
        cnts = input_data[:, -1, :]
        
        # Apply embedding to pts
        x = self.embed(pts.view(batch_size, -1))
        x = self.linears(x)
        
        if self.short_cut:
            embedded_cnts = self.embed(cnts)
            x = torch.cat([x, embedded_cnts], dim=1)

        weight = self.linears1(x)

        return weight


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [4]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)