import os
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import math
import random
import copy
from PIL import Image
import nibabel as nib

import torch
from torch.utils.data import Dataset

from scipy.ndimage import gaussian_filter
from numpy.linalg import eig

import torch.nn.functional as F

from src import utils

from scipy.ndimage import gaussian_filter
from numpy.linalg import eig

from tqdm import tqdm

class Base(Dataset):
    def __init__(self, params):
        super(Base, self).__init__()
        for key, value in params.items():
            setattr(self, key, value)

    def __getitem__(self, index):
        return self.single_view_sampling(index) 

    def __len__(self):
        return self.LEN

    def setup(self):
        if self.mode == 'train':
            self.LEN = self.len
            self.pad = self.radius
            #这里self.radius=1

        elif self.mode == 'eval' or self.mode == 'test':
            
            self.vals = list(range(self.len * self.scale))
            self.LEN = self.len * self.scale
            #这里没有放缩
            self.pad = self.radius


        else:
            print (f'Not {self.mode} mode!')
            exit()

        self.pad = min(self.pad, 1)

    def z_trans(self, z):
        return 2 * np.pi * (z + self.pad) / (self.LEN + 2 * self.pad - 1) - np.pi
        #这里修改了
    
    def sampling(self, coords, xy_inds, z_coord, pad):
        #xy_inds只是用来提取index的，所有点都是在coords(-pi,pi)中选择的，
        #这一步相当于是1:将坐标中心的变换到图像正中心，2:embedding
        
        # 获取采样点的 xy 坐标
        xy_coords = coords[xy_inds[:, 0] + pad, xy_inds[:, 1] + pad]
        # 获取采样点的左右边界坐标
        LR_coords = torch.cat([coords[xy_inds[:, 0] + pad, xy_inds[:, 1] + ind][:, 0:1] for ind in [0, pad * 2]], 1)
        # 获取采样点的上下边界坐标
        TB_coords = torch.cat([coords[xy_inds[:, 0] + ind, xy_inds[:, 1] + pad][:, 1:2] for ind in [0, pad * 2]], 1)
        # 将所有坐标信息组合在一起，形成 (n, 7) 的形状
        coords = torch.cat([xy_coords, torch.full((xy_coords.shape[0], 1), z_coord), LR_coords, TB_coords], 1)
        return coords


    def random_sample(self, coords):
        """
        随机在3D空间内采样 bsize 个点，返回对应的坐标和数据。
        """
        # 随机选择 bsize 个 z, x, y 索引，确保不越界
        z_inds = torch.randint(self.pad, self.len - self.pad, (self.bsize,))
        x_inds = torch.randint(0, self.H, (self.bsize,))
        y_inds = torch.randint(0, self.W, (self.bsize,))

        # 将 xy_inds 组合
        xy_inds = torch.stack([x_inds, y_inds], dim=1)
        #print("xy_inds", xy_inds.shape)
        #print("z_inds", z_inds.shape)
        
        #print("x_inds, y_inds, z_inds",x_inds[0], y_inds[0], z_inds[0])
        #print("torch.stack([[x_inds[0]], [y_inds[0]]], dim=1)",torch.tensor([[x_inds[0], y_inds[0]]]).shape,torch.tensor([[x_inds[0], y_inds[0]]]))
        demo=self.sampling(coords, torch.tensor([[x_inds[0], y_inds[0]]]), z_coord=torch.tensor(0, dtype=torch.float32), pad=self.pad)

        #print("demo",demo, torch.tensor([z_inds[0]]).shape,self.z_trans(torch.tensor([z_inds[0]])))
        

        # 处理前 7 个坐标值
        coords_2d = self.sampling(coords, xy_inds, z_coord=torch.tensor(0, dtype=torch.float32), pad=self.pad)  # z_coord 先设为 None


        # 现在为每个采样点添加 z 坐标
        z_coords = self.z_trans(z_inds)  # shape: [bsize]

        # 计算 z-, z+
        z_min = self.z_trans(z_inds - self.pad)
        z_max = self.z_trans(z_inds + self.pad)

        # 将 z_coord 填充到前 7 个坐标中
        coords_2d[:, 2] = z_coords  # 替换 z_coord 值

        # 组合成 coords，形状为 [bsize, 9]
        coords = torch.cat([
            coords_2d,               # [bsize, 7]
            z_min.unsqueeze(1),      # [bsize, 1]
            z_max.unsqueeze(1)       # [bsize, 1]
        ], dim=1)  # shape: [bsize, 9]

        # 获取采样点的数据值
        data = self.data[z_inds, x_inds, y_inds]

        return coords, data

    def single_view_sampling(self, index):
        # 创建一个二维网格，用于表示图像的坐标
        j, i = torch.meshgrid(torch.linspace(-np.pi, np.pi, self.H + 2 * self.pad), torch.linspace(-np.pi, np.pi, self.W + 2 * self.pad))
        # 将坐标堆叠成 (H + 2*pad, W + 2*pad, 2) 的形状，包含每个点的 (i, j) 坐标
        coords = torch.stack([i, j], -1)

        if self.mode == 'train':
            # 在训练模式下，从低分辨率网格中选择采样点
            xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H), torch.linspace(0, self.W - 1, self.W))
            # 计算 z 轴上的索引，根据缩放比例选择合适的 z 索引
            z_ind = index
            # 将二维网格展平成 (n, 2) 的形状，n 是采样点的数量

            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            # 如果采样点数量超过 bsize，则随机选择 bsize 个采样点
            if xy_inds.shape[0] > self.bsize:
                xy_inds_1024 = xy_inds[np.random.choice(xy_inds.shape[0], size=[self.bsize], replace=False)]
                #xy_inds_1024 = xy_inds[np.random.choice(xy_inds.shape[0], size=[4096], replace=False)]
                #xy_inds_1024=xy_inds
        
            # 对 z 轴索引进行归一化转换，变换到 [-π, π] 区间
            head, z_coord, tail = [self.z_trans(z) for z in [z_ind - self.pad, z_ind, z_ind + self.pad]]
            coords_full = self.sampling(coords, xy_inds, z_coord, self.pad)
        
            #xy_inds只是用来提取index的，所有点都是在coords(-pi,pi)中选择的，
            #这一步相当于是1:将坐标中心的变换到图像正中心，2:embedding到(-pi,pi)
            coords = self.sampling(coords, xy_inds_1024, z_coord, self.pad)

            data_full = self.data[z_ind, xy_inds[:, 0], xy_inds[:, 1]]

            # 获取采样点在高分辨率图像中的对应像素值
            data = self.data[z_ind, xy_inds_1024[:, 0], xy_inds_1024[:, 1]]

            #coords, data = self.random_sample(coords)
        
            data_hessian_full = self.hessian_processed[z_ind, xy_inds[:, 0], xy_inds[:, 1]]
            data_merge=(data, data_full, data_hessian_full)

            head = torch.tensor(head, dtype=torch.float32)
            tail = torch.tensor(tail, dtype=torch.float32)

            head_tensor = head.view(1, 1)
            tail_tensor = tail.view(1, 1)

            # 调整 head_tensor 和 tail_tensor 的形状
            head_tensor1 = head_tensor.expand(coords.shape[0], -1)  # 形状变为 [1024, 1]
            tail_tensor1 = tail_tensor.expand(coords.shape[0], -1)  # 形状变为 [1024, 1]
            
            head_tensor2 = head_tensor.expand(coords_full.shape[0], -1)  
            tail_tensor2 = tail_tensor.expand(coords_full.shape[0], -1) 

            # 现在可以安全地连接
            coords = torch.cat((coords, head_tensor1, tail_tensor1), dim=1)  # 结果形状 [1024, 9]
            coords_full = torch.cat((coords_full, head_tensor2, tail_tensor2), dim=1)  # 结果形状 [1024, 9]
            
            return (data_merge, coords, coords_full, self.data.shape, self.scale)
            #return (data_merge, coords, coords_full, (np.float32(head), np.float32(tail), self.data.shape, self.scale)
        

        else:
            # 在评估或测试模式下，使用完整的高分辨率网格
            xy_inds = torch.meshgrid(torch.linspace(0, self.H - 1, self.H * self.scale), torch.linspace(0, self.W - 1, self.W * self.scale))
            # 将二维网格展平成 (n, 2) 的形状，n 是采样点的数量
            xy_inds = torch.stack(xy_inds, -1).reshape([-1, 2]).long()
            # 选择对应的 z 索引
            z_ind = self.vals[index]
        

            # 对 z 轴索引进行归一化转换，变换到 [-π, π] 区间
            head, z_coord, tail = [self.z_trans(z) for z in [z_ind - self.pad, z_ind, z_ind + self.pad]]
            coords_full = self.sampling(coords, xy_inds, z_coord, self.pad)
        
            #xy_inds只是用来提取index的，所有点都是在coords(-pi,pi)中选择的，
            #这一步相当于是1:将坐标中心的变换到图像正中心，2:embedding到(-pi,pi)
            coords = self.sampling(coords, xy_inds, z_coord, self.pad)

        

            # 返回采样的像素值、坐标和 z 轴范围信息
            return (coords, (np.float32(head), np.float32(tail)), self.data.shape, self.scale)


class Medical3D(Base):
    def __init__(self, **params):
        super(Medical3D, self).__init__(params)
        self.load_data()
        if self.mode == 'test':
            self.angles = np.linspace(self.angles[0], self.angles[1], self.asteps) if len(self.angles) == 2 else [self.angles[0]] * self.asteps
            self.scales = np.linspace(self.scales[0], self.scales[1], self.asteps) if len(self.scales) == 2 else [self.scales[0]] * self.asteps
            self.zs = np.linspace(self.zpos[0], self.zpos[1], self.asteps) if len(self.zpos) == 2 else [self.zpos[0]] * self.asteps

    def load_data(self):
        print("def load_data(self):")
        data = self.load_file()
        #data = self.nomalize(data)
        #self.data = self.align(data)

        if self.mode=='train':
        
            raw_data = self.align(data.clone()).cpu().numpy()
            
            #nifti_image = nib.Nifti1Image(raw_data, affine=np.eye(4))

            #nib.save(nifti_image, '/data/zhaolab/zcm/learning_nerf/CuNeRF/raw_data.nii.gz')
            
            ###
            hessian = self.hessian_enhance_3d(raw_data)
            hessian[hessian>0]=255
            raw_data[hessian==0]=0
            hessian_processed=gaussian_filter(raw_data,2)
            ###

            #img_hessian = nib.load('G:/VideoINR/draft/code/CuNeRF/data/test.nii.gz')
            #hessian_processed = img_hessian.get_fdata()
            
            
            #nifti_image = nib.Nifti1Image(hessian_processed, affine=np.eye(4))
            #nib.save(nifti_image, '/data/zhaolab/zcm/learning_nerf/CuNeRF/hessian_processed.nii.gz')
            hessian_processed=self.nomalize(hessian_processed)
            
            #hessian_processed=self.nomalize(hessian)
            
            self.hessian_processed = torch.from_numpy(hessian_processed).float().cuda()
        else:
            data1 = self.nomalize(data.clone())
            self.hessian_processed = self.align(data1)
        
        data = self.nomalize(data)
        self.data = self.align(data)
        self.len, self.H, self.W = self.data.shape
        print (self.len, self.H, self.W)
        self.setup()
        print("def load_data(self): done")

    def align(self, data):
        if data.shape[1] != data.shape[2]:
            if data.shape[0] == data.shape[2]:
                data = data.permute(1, 0, 2)

            else:
                data = data.permute(2, 1, 0)
        
        return data


    def hessian_enhance_3d(self, line_stack):
        sr, sc, sz = line_stack.shape
    
        gau_stack = gaussian_filter(line_stack.astype(float), 2)
    
        # 计算平滑图像的梯度
        ix, iy, iz = np.gradient(gau_stack)
    
        # 计算二阶导数（Hessian 矩阵的组成部分）
        ixx, ixy, ixz = np.gradient(ix)
        iyx, iyy, iyz = np.gradient(iy)
        izx, izy, izz = np.gradient(iz)
    
        # 初始化 Hessian 特征值矩阵
        ihessian1 = np.zeros((sr, sc, sz))
        ihessian2 = np.zeros((sr, sc, sz))
        ihessian3 = np.zeros((sr, sc, sz))
    
        # 找到平滑图像中非零元素的索引
        indx_valid = np.nonzero(gau_stack)
        x_indx, y_indx, z_indx = indx_valid
    
        # 处理每个不等于0的点
        for i, j, k in tqdm(zip(x_indx, y_indx, z_indx)):
            # 在每个点构建 Hessian 矩阵
            hessia_matrix = np.array([
                [ixx[i, j, k], ixy[i, j, k], ixz[i, j, k]],
                [iyx[i, j, k], iyy[i, j, k], iyz[i, j, k]],
                [izx[i, j, k], izy[i, j, k], izz[i, j, k]]
            ])
        
            # 计算 Hessian 矩阵的特征值
            e_value = eig(hessia_matrix)[0]
        
            # 按特征值的绝对值排序
            e_value = e_value[np.argsort(np.abs(e_value))]
        
            # 将排序后的特征值分配给相应的矩阵
            ihessian1[i, j, k] = e_value[0]
            ihessian2[i, j, k] = e_value[1]
            ihessian3[i, j, k] = e_value[2]
    
        # 使用第三个特征值增强膜图像
        h_filtered_mem = np.abs(np.minimum(ihessian3, np.zeros((sr, sc, sz))))
    
        # 对增强后的图像应用阈值
        h_filtered_mem = (h_filtered_mem > 0.2) * h_filtered_mem
        #h_filtered_mem = gaussian_filter(h_filtered_mem, 2)
        
        return h_filtered_mem



    def load_file(self):
        data = sitk.GetArrayFromImage(sitk.ReadImage(self.file)).astype(float)
        #啥啊这是，为啥不用nib？import SimpleITK as sitk，
        data = torch.from_numpy(data).float().cuda()
        if len(data.shape) == 4:
            modalities = {
                'FLAIR' : 0,
                'T1w'   : 1,
                't1gd'  : 2,
                'T2w'   : 3
            }
            data = data[modalities[self.modality]]
        return data

    def nomalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def getLabel(self):
        return self.data[self.vals].cpu().numpy()