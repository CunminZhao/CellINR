import torch
import torch.nn.functional as F

import math

def cube_rendering(raw, pts, cnts, dx, dy, dz):
    # norm 计算每一个采样点到中心点的距离的范数
    norm = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz))
    
    # 定义将 raw 值转换为 beta 和 alpha 的函数
    raw2beta = lambda raw, dists, rs, act_fn=F.relu : -act_fn(raw) * dists * torch.square(rs) * 4 * math.pi
    raw2alpha = lambda raw, dists, rs, act_fn=F.relu : (1.-torch.exp(-act_fn(raw) * dists)) * torch.square(rs) * 4 * math.pi 
    
    # 计算每个点到中心点的欧几里得距离
    rs = torch.norm(pts - cnts[:, None], dim=-1)
    sorted_rs, indices_rs = torch.sort(rs)
    
    # 计算每个采样点之间的距离
    dists = sorted_rs[...,1:] - sorted_rs[...,:-1]
    dists = torch.cat([dists, dists[...,-1:]], -1)  # [N_rays, N_samples]

    # 按照距离对颜色和密度进行排序
    rgb = torch.gather(torch.sigmoid(raw[...,:-1]), -2, indices_rs[..., None].expand(raw[...,:-1].shape))
    sorted_raw = torch.gather(raw[...,-1], -1, indices_rs)

    # 将 raw 值转换为 beta 和 alpha
    beta = raw2beta(sorted_raw, dists, sorted_rs / norm)
    alpha = raw2alpha(sorted_raw, dists, sorted_rs / norm)  # [N_rays, N_samples]

    # 计算权重
    weights = alpha * torch.exp(torch.cumsum(torch.cat([torch.zeros(alpha.shape[0], 1), beta], -1), -1)[:, :-1])

    # 计算最终的颜色
    rgb_map = torch.sum(weights * rgb.squeeze(), -1)

    return {'rgb' : rgb_map, 'weights' : weights, 'indices_rs' : indices_rs}