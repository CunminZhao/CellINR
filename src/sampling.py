import torch
import math

def cube_sampling(batch, n_samples, is_train, R=None):
    # 解析输入的 batch，将其拆分为 cnts, LR, TB, near 和 far
    #print(batch.shape)
    #print("BATCH")
    (cnts, LR, TB, near, far) = torch.split(batch, [3, 2, 2, 1, 1], dim=-1)

    # 将 near 和 far 从 (batch_size, 1) 形状变为 (batch_size, n_samples) 形状
    near = near.expand(-1, n_samples)
    far = far.expand(-1, n_samples)

    # 将 LR 和 TB 切分为左右边界和上下边界
    left, right = torch.split(LR, [1, 1], dim=-1)
    top, bottom = torch.split(TB, [1, 1], dim=-1)

    # 计算采样点的位置
    steps = int(math.pow(n_samples, 1./3) + 1)
    
    t_vals = torch.cat([v[..., None] for v in torch.meshgrid(
        torch.linspace(0., 1., steps=steps),
        torch.linspace(0., 1., steps=steps),
        torch.linspace(0., 1., steps=steps)
    )], -1)
    t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3)

    x_l, x_r = left.expand(-1, n_samples), right.expand(-1, n_samples)
    y_l, y_r = top.expand(-1, n_samples), bottom.expand(-1, n_samples)

    # 计算采样点的 x, y, z 坐标
    if is_train:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand_like(x_l)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand_like(y_l)
        z_vals = near + t_vals[:, 2] * (far - near) * torch.rand_like(near)
    else:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
        z_vals = near + t_vals[:, 2] * (far - near)

    # 拼接采样点的坐标
    pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)

    # 如果提供了旋转矩阵 R，则应用它
    if R is not None:
        pts, cnts = pts @ R, cnts @ R

    return {
        'pts': pts,
        'cnts': cnts,
        'dx': (x_r - x_l).mean() / 2,
        'dy': (y_r - y_l).mean() / 2,
        'dz': (far - near).mean() / 2
    }


#这段代码实现了一种改进版的采样方法，用于神经辐射场（NeRF）。具体来说，它在一个三维空间中采样点，用于训练或推理阶段。以下是代码的整体流程和功能：

#输入参数：
#batch：包含一批次的采样数据，维度为 [batch_size, 7]。每一行包含一个采样中心点（3维）和左右、上下边界（各1维），即 cnts, LR, TB。
#depths：包含每个采样点的最近和最远深度值，即 near 和 far。
#n_samples：采样点的数量。
#is_train：标记是否为训练阶段。
#R：旋转矩阵，用于旋转采样点。
#数据拆分：
#将 batch 拆分为中心点 cnts、左右边界 LR 和上下边界 TB。
#将 depths 拆分为最近和最远深度值 near 和 far。
#生成采样网格：
#使用 torch.meshgrid 创建一个三维网格 t_vals，用于在每个维度上均匀采样。
#扩展边界值：
#扩展左右、上下边界和深度值，使其与采样点数量匹配。
#采样点计算：
#根据 is_train 标志，决定是随机采样还是均匀采样：
#训练阶段：在每个维度上随机采样。
#推理阶段：在每个维度上均匀采样。
#组合采样点：
#将采样的 x、y、z 坐标组合成最终的三维采样点 pts。
#旋转采样点（如果提供了旋转矩阵 R）：
#使用旋转矩阵旋转采样点和中心点。
#返回结果：
#返回一个包含采样点 pts、中心点 cnts，以及每个维度上的平均步距的字典。
#总体而言，这段代码的功能是根据给定的边界和深度信息，在三维空间中采样点，并返回这些点及相关信息，用于后续的神经辐射场计算中。