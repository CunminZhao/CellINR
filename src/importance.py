import torch
import math

def cube_imp(weights, indices_rs, pts, cnts, is_train, n_samples, **kwargs):
    # 这个函数用于改进采样点的选择，通过使用先前计算的权重来选择更有效的点
    # weights: 每个采样点的权重
    # indices_rs: 采样点的索引
    # pts: 当前采样点的坐标
    # cnts: 立方体的中心点坐标
    # is_train: 是否在训练模式
    # n_samples: 需要采样的点数

    # 使用 cube_sample_pdf 函数根据权重来选择新的采样点
    pts_rs = cube_sample_pdf(pts, cnts, weights[..., 1:-1], indices_rs, n_samples, is_train).detach()
    # 将新选择的采样点与旧的采样点结合在一起
    pts = torch.cat([pts, pts_rs + cnts[:, None]], 1)
    return {'pts' : pts, 'cnts' : cnts, 'dx' : kwargs['dx'], 'dy' : kwargs['dy'], 'dz' : kwargs['dz']}

def cube_sample_pdf(pts, cnts, weights, indices_rs, N_samples, is_train):
    # 这个函数根据给定的权重和坐标选择新的采样点
    # pts: 当前采样点的坐标
    # cnts: 立方体的中心点坐标
    # weights: 采样点的权重
    # indices_rs: 采样点的索引
    # N_samples: 需要采样的点数
    # is_train: 是否在训练模式

    # 计算每个点到中心的距离
    centers = torch.gather(pts - cnts[:, None], -2, indices_rs[..., None].expand(*pts.shape)) 
    mids = .5 * (centers[:, 1:] + centers[:, :-1]) # 计算中点
    rs_mid = torch.norm(mids, dim=-1) # 计算中点的欧几里得距离
    # xs_mid, ys_mid, zs_mid = mids[...,0], mids[...,1], mids[...,2]
    weights = weights + 1e-5 # 避免除以零
    pdf = weights / torch.sum(weights, -1, keepdim=True) # 计算概率密度函数
    cdf = torch.cumsum(pdf, -1) # 计算累积分布函数
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1) # 添加零以便计算
    # Take uniform samples
    # 在训练模式下随机选择样本，否则均匀选择
    if is_train:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    
    else:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True) # 查找索引
    below = torch.max(torch.zeros_like(inds-1), inds-1) # 找到低于的索引
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # 找到高于的索引
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2) # 组合低高索引
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    # 获取对应的中点和权重
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) 
    bins_g = torch.gather(rs_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[...,1] - cdf_g[...,0]  # 计算分母
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)  # 避免分母为零
    t = (u - cdf_g[...,0]) / denom  # 计算插值

    # 生成新的采样点
    rs = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    ts = torch.rand_like(rs) * math.pi
    ps = torch.rand_like(rs) * 2 * math.pi

    xs = rs * torch.sin(ts) * torch.cos(ps) # x坐标
    ys = rs * torch.sin(ts) * torch.sin(ps) # y坐标
    zs = rs * torch.cos(ts) # z坐标
    samples = torch.cat([xs[...,None], ys[...,None], zs[...,None]], -1) # 合并坐标

    return samples # 返回新的采样点