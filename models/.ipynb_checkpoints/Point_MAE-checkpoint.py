import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from torch.autograd import Function
from typing import Tuple

def resort_points(points, idx):

    device = points.device
    B, N, G, _ = points.shape

    view_shape = [B, 1, 1]
    repeat_shape = [1, N, G]
    b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points

def check_nan_umb(normal, center, pos=None):
    B, N, G, _ = normal.shape
    mask = torch.sum(torch.isnan(normal), dim=-1) > 0
    mask_first = torch.argmax((~mask).int(), dim=-1)
    b_idx = torch.arange(B).unsqueeze(1).repeat([1, N])
    n_idx = torch.arange(N).unsqueeze(0).repeat([B, 1])

    normal_first = normal[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    normal[mask] = normal_first[mask]
    center_first = center[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
    center[mask] = center_first[mask]

    if pos is not None:
        pos_first = pos[b_idx, n_idx, None, mask_first].repeat([1, 1, G, 1])
        pos[mask] = pos_first[mask]
        return normal, center, pos
    return normal, center

def cal_const(normal, center, is_normalize=True):
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const

def cal_center(group_xyz):
    center = torch.mean(group_xyz, dim=-2)
    return center

def xyz2sphere(xyz, normalize=True):
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out

def cal_normal(group_xyz, random_inv=False, is_group=False):
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3]
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    if random_inv:
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return unit_nor

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(10, 128, 1),# 3--.10
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, 1)
        # )
        self.first_conv=nn.Conv1d(10,256,1)
        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, self.encoder_channel, 1) #384
        # )
        self.second_conv=nn.Conv1d(512,encoder_channel,1)
    def forward(self, point_groups):
        '''
            point_groups : B G N 10
            -----------------
            feature_global : B G C
        '''
        #print("Stephanie")
        bs, g, n , _ = point_groups.shape #bs=b g=64(center) n=32(k) num_featrues=10
        point_groups = point_groups.reshape(bs * g, n, 10) #(2048, 32,10)  #3--10 (4*128, 32,3)
        # encoder   point_groups.transpose(2,1)=(2048, 10, 32 )
        feature = self.first_conv(point_groups.transpose(2,1)) #(b*center, 256, 32) # BG 256 n #nan
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  #(b*center, 256, 1)  # BG 256 1
        temp=feature_global.expand(-1,-1,n); #(b*center, 256, 32) n=32
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)  #(b*center, 512, 32)  # BG 512=256*2 n #返回当前张量在某维扩展更大后的张量
        feature = self.second_conv(feature) #(b*center, 384, 32)  #512,384,32 BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] #(b*center, 384)  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel) #(b, 64, 384)#4 128 384
    
class Encoder2(nn.Module):
    def __init__(self,encoder_channel):
        super().__init__();
        self.encoder_channel = encoder_channel
        self.first_conv=nn.Conv1d(10,256,1)
        self.second_conv=nn.Conv1d(512,encoder_channel,1)
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        #print("Stephanie")
        bs, g, n , _ = point_groups.shape #bs=b g=64(center) n=32(k) num_featrues=10
        point_groups = point_groups.reshape(bs * g, n, 10) #(2048, 32,10)  #3--10 (4*128, 32,3)
        # encoder   point_groups.transpose(2,1)=(2048, 10, 32 )
        feature = self.first_conv(point_groups.transpose(2,1)) #(b*center, 256, 32) # BG 256 n #nan
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  #(b*center, 256, 1)  # BG 256 1
        temp=feature_global.expand(-1,-1,n); #(b*center, 256, 32) n=32
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)  #(b*center, 512, 32)  # BG 512=256*2 n #返回当前张量在某维扩展更大后的张量
        feature = self.second_conv(feature) #(b*center, 384, 32)  #512,384,32 BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] #(b*center, 384)  # BG 1024
        temp=feature_global.reshape(bs, g, self.encoder_channel) #(b, 64, 384)#4 128 384
        return temp

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, is_print=False):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        if is_print:
            print("input point:",xyz);
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        if is_print:
            print("input center:",center);
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        if is_print:
            print("chose idx:",idx);
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1);
        if is_print:
            print("after transforms idx:",idx);
        temp=xyz.view(batch_size * num_points, -1) #(1*2048,3) original input points
        if is_print:
            print("input temp:",temp[0,]);
            print("input temp:",temp[92,]);
            print("input temp:",temp[99,]);
            
        neighborhood = temp[idx, :]
        neighborhood=xyz.view(batch_size * num_points, -1)[idx,:]
        
        if is_print:
            print("before transform neighbor:",neighborhood);
       
        neighborhood = neighborhood.contiguous().view(batch_size, self.num_group, self.group_size, 3)
        if is_print:
            print("chose neighbor:",neighborhood);
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        if is_print:
            print("after minus neighbor:",neighborhood);
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        #Point-MAE进来的是masked_center(4,26,3)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), #linear是变最后一个dim (4,26,128)
            nn.GELU(),
            nn.Linear(128, self.trans_dim), #(4,26,384)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim) #trans_dim=384
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False): #neighborhood=new_feature (b,64,32,10)
        # generate mask
        #print("hello!mask!")
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G #(4,64)里面都是true/false num_group: 64
        else:
            #print("hello!mask!")
            bool_masked_pos = self._mask_center_block(center, noaug = noaug) #(4,64)，里面value都是T/F，表明哪个patch需要/不需要被mask
        #print("hello!mask!")
        group_input_tokens = self.encoder(neighborhood)  #(b, 64, 384) 这个384是10变的 #(4,64,384) encoder_dims: 384 #  B G C

        batch_size, seq_len, C = group_input_tokens.size() #batch_size:4, seq_len:64, C:384

        #group_input_tokens[~bool_masked_pos]:(104,384)，4个点云里一共有104个可见的patch
        #group_input_tokens[bool_masked_pos]:(152,384)  #104+152=256=4*64
        #bool_masked_pos中，152个true，104个false
        x_vis_input = group_input_tokens[~bool_masked_pos] #4*8=32 (832,384)  #(104,384) 104可见，数字不变的
        x_vis = x_vis_input.reshape(batch_size, -1, C) #(32, 26, 384) -1=(832*384)/(32*384)=26 #(4,26,384), 每个点云里有26个可见的patch， 26=104/4
        # add pos embedding
        # mask pos center
        vis_center = center[~bool_masked_pos] #(832, 3) #(104,3) center原本(4,64,3)
        masked_center = vis_center.reshape(batch_size, -1, 3) #(32,26,3)#(4,26,3)
        pos = self.pos_embed(masked_center) #(4,26,384) 这个384是3变的

        # transformer
        x_vis = self.blocks(x_vis, pos) #(32,26,384) (32,26,384) #(4,26,384)
        x_vis = self.norm(x_vis) #(32,26,384) #(4,26,384) #LayerNorm

        return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim)) #(1,1,384)
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads


        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.surfacemlps = nn.Sequential(
            #nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, 1, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 10, 1, bias=True),
        )

        #the following attribute is different from transformer.
        self.MAE_encoder = MaskTransformer(config)

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.decoder_pos_embed = nn.Sequential(  # MLP
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim, #384
            depth=self.decoder_depth, #4
            drop_path_rate=dpr, #0.1
            num_heads=self.decoder_num_heads, #6
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 320, 1) #in:384 out=32*3=96  nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        #for m in self.modules():
            #if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            #elif isinstance(m, nn.BatchNorm2d):
                #nn.init.constant_(m.weight, 1)
                #nn.init.constant_(m.bias, 0)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':#this way
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, pts, vis = False, **kwargs):
        #center (4,64,3)
        #neighborhood ()
        neighborhood, center = self.group_divider(pts)
        polar_feature = xyz2sphere(neighborhood) #(4,64,32,3)
        group_phi = polar_feature[..., 2]  #(4,64,32) # [B, N', K-1]
        sort_idx = group_phi.argsort(dim=-1)  #(4,64,32) # [B, N', K-1]

        # [B, N', K-1, 1, 3]
        sorted_group_xyz = resort_points(neighborhood, sort_idx).unsqueeze(-2) #(4,64,32,1,3)
        sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3) #(4,64,32,1,3)
        group_centriod = torch.zeros_like(sorted_group_xyz) #(4,64,32,1,3)
        umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2) #(4,64,32,3,3) polar,xyz
        group_normal = cal_normal(umbrella_group_xyz, random_inv=True, is_group=True) #(4,64,32,3) normal vector
        group_center = cal_center(umbrella_group_xyz) #(4,64,32,3) 求mean
        group_polar = xyz2sphere(group_center) #(4,64,32,3)
        group_pos = cal_const(group_normal, group_center) #(4,64,32,1)
        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos)

        new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1) #(b,64,32,10) # N+P+CP: 10

        new_feature = new_feature.permute(0, 3, 2, 1) #(4, 10, 32, 64) channel first

        new_feature = self.surfacemlps(new_feature);


        new_feature = new_feature.permute(0, 3, 2, 1)
        print("MAE_encoder")
        x_vis, mask = self.MAE_encoder(new_feature, center) #input(b,64,32,10) (b,64,3) output:(b,26,384) (32 64)
        B,_,C = x_vis.shape # B _=26 C=384

        # center[~mask] (104.3)
        # self.decoder_pos_embed(center[~mask]) (104,384)
        temp1= self.decoder_pos_embed(center[~mask]) #(832, 384)
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)  #(b, 26, 384)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C) #(b, 38, 384) #masked

        _,N,_ = pos_emd_mask.shape #N=38
        mask_token = self.mask_token.expand(B, N, -1) #(b, 38, 384)
        x_full = torch.cat([x_vis, mask_token], dim=1) #(b, 64, 384)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1) #(b, 64, 384)

        x_rec = self.MAE_decoder(x_full, pos_full, N) #(b, 38, 384) transformer

        B, M, C = x_rec.shape   #b M=38 384
        #x_rec(32,38,384)-(32,96,38)-(32,38,96)-
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 10)  #-1的位置必须是32 # B M 1024 #(1216, 32, 3)  .reshape(B * M, -1, 3)

        gt_points = new_feature[mask].reshape(B*M,-1,10) #.reshape(B*M,-1,3) #(1216,32,10)

        loss1 = self.loss_func(rebuild_points, gt_points) #倒角距离

        if vis: #visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss1

# finetune model
class MLP(nn.Module):
    def __init__(self,):
        self.conv1=nn.Conv2d(in_channels=10, out_channels=5, kernel_size=1, bias=False)
        self.bn1=nn.BatchNorm2d(5)
        self.conv2=nn.Conv2d(in_channels=5, out_channels=10, kernel_size=1, bias=False)
        self.bn2=nn.BatchNorm2d(10)
        self.activate=nn.ReLU(True)
    def forward(self,x):
        x=self.conv1(x);
        x=self.bn1(x);
        x=self.activate(x);
        x=self.conv2(x);
        x=self.bn2(x);
        return x
        

@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        self.surfacemlps = nn.Sequential(
             # nn.BatchNorm2d(10),
             #nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, bias=False),
             #nn.BatchNorm2d(10),
             nn.ReLU(True),
             #nn.LeakyReLU(0.01),
             #nn.Conv2d(in_channels=5, out_channels=10, kernel_size=1, bias=False),
             #nn.BatchNorm2d(10),
             #nn.ReLU(True),
             # nn.Conv2d(10, 10, 1, bias=True),
         )
        #self.surfacemlps=MLP();
        # the following content is different from MAE
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.pos_embed = nn.Sequential(  # MLP
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim)) #(1,1,384) transformer
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim)) #(1,1,384) transformer


        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim, #384
            depth=1, #self.depth, #12
            drop_path_rate= dpr, #0.1
            num_heads=self.num_heads #6
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.cls_head_finetune = nn.Sequential(
        #         nn.Linear(self.trans_dim * 2, 256),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, 256),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, self.cls_dim)
        #     )
        self.cls_head_finetune=nn.Linear(self.trans_dim,self.cls_dim)
        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)


    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt): #vector score
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                print("k",k)# k is key str
                if k.startswith('MAE_encoder') :
                    temp=len('MAE_encoder.')
                    temp2=k[len('MAE_encoder.'):]
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')

        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=1)#0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            trunc_normal_(m.weight, std=1)#0.02
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, num=5):
        torch.autograd.set_detect_anomaly(True)
        #if num==5 or num==6:
            #print("here____________")
        neighborhood, center = self.group_divider(pts);
        #print("center shape",center.shape) #(bs, num_group, 3)
        #print("here____________")
        ########################################################################
        # transplant from  surface
        polar_feature = xyz2sphere(neighborhood)
        group_phi = polar_feature[..., 2]  # [B, N', K-1]
        sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

        # [B, N', K-1, 1, 3]
        sorted_group_xyz = resort_points(neighborhood, sort_idx).unsqueeze(-2)
        sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
        group_centriod = torch.zeros_like(sorted_group_xyz)
        umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)
        group_normal = cal_normal(umbrella_group_xyz, random_inv=True, is_group=True)
        group_center = cal_center(umbrella_group_xyz)
        group_polar = xyz2sphere(group_center)
        group_pos = cal_const(group_normal, group_center)
        group_normal, group_center, group_pos = check_nan_umb(group_normal, group_center, group_pos) #check 0 infinit smaler value
    
        #print("here____________")
        new_feature = torch.cat([group_center, group_polar, group_normal, group_pos], dim=-1)  # N+P+CP: 10
        #new_feature:(batchsize, n_group=128, group_size=32, num_channel=10) (4, 128, 32, 10)
        print("new feature:", new_feature);
        if torch.any(torch.isnan(new_feature)):
            bs, n_group,group_size,num_channel = new_feature.shape
            #for group_no in range(n_group): 
            print("neighborhood:",neighborhood[0,0,:])
            print("center:",center[0,0,:])
            print("problem channel:",new_feature[0,0,:,:])
            self.group_divider(pts,True)
    
        #new feature no Nan
        new_feature = new_feature.permute(0, 3, 2, 1)
        # (4, 10, 32, 64) channel first
        #(bs,num_group, group_size, num_channel)-->(bs,num_channel, group_size, num_group)
        
        
            
            
        new_feature=-new_feature;
        #print("new feature:", new_feature[0][0][0]);
        new_feature = self.surfacemlps(new_feature) #Nan
        print("after surface:", new_feature)
        new_feature = new_feature.permute(0, 3, 2, 1);
        # #(bs,num_channel, group_size, num_group)-->(bs, num_group, group_size, num_channels)
        #########################################################################################
        group_input_tokens = self.encoder(new_feature)  # B G N
        #(bs, num_group, group_size, num_channels)-->(bs,num_group, 384)
        print("after encoder:",  group_input_tokens[0][0])
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)#(1,1,384) to (4,1,384)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)
        #print(pos.shape);

        x = torch.cat((cls_tokens, group_input_tokens), dim=1) #transformer
        pos = torch.cat((cls_pos, pos), dim=1) #transformer
        
        # transformer
        x = self.blocks(x, pos) #(4,384)
        #print("after transformer",x);
        x = self.norm(x) #(4,129,384)
        #print("after norm", x.shape)
        #concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1);
        concat_f=x[:,0] # (bs, 384)
        #print(concat_f.shape)
        print("before classifier:", concat_f);
        #assert not torch.any(torch.isnan(concat_f))
        ret = self.cls_head_finetune(concat_f) #classifier #(bs,384)-->(bs,15)
        
        #print(ret.shape);
        print("score_vector:",ret);
        assert not torch.any(torch.isnan(ret))
        return ret
