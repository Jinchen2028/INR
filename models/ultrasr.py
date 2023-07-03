import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import register
from utils import make_coord
from tool.func import PosEncoding_sin


@register('ultrasr')
class UltraSR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=False,
                 in_features=2, sidelength=512, fn_samples=None, use_nyquist=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold  # 展开操作可以增加特征的感受野，从而更好地捕获图像的上下文信息
        self.cell_decode = cell_decode  # 解码操作可以使模型在预测时考虑单元格的大小和形状
        self.positional_encoding = PosEncoding_sin(in_features=in_features,
                                               sidelength=sidelength,
                                               fn_samples=fn_samples,
                                               use_nyquist=use_nyquist)

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:  # 未提供该参数，模型只能进行特征提取，而无法预测RGB值，把EDSR的输出给mlp
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2  # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            if self.positional_encoding:
                imnet_in_dim += 30
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})  # 建立mlp
        else:
            self.imnet = None
        # self.pos_encode = SineLayer(in_features=2, out_features=2)  # 直接输入到隐藏层

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat  # 它接受输入图像 inp，并通过编码器将其转换为特征表示。生成的特征将在后续的查询过程中使用

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)  # 采样，并将结果进行转置和维度变换
            return ret

        if self.feat_unfold:  # 对feat进行展开，将每个像素周围3x3的区域展开为一维向量
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:  # 偏移列表
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # 创建特征图坐标张量

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # 据偏移量进行坐标调整
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # 对调整后的坐标在特征图上进行网格采样，得到查询点的特征表示和坐标
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # 乘以特征图的尺寸，得到相对于特征图尺寸的坐标偏移量。
                rel_coord_ = self.positional_encoding(rel_coord)
                rel_coord_ = torch.cat([rel_coord, rel_coord_], dim=-1)
                # print('rel_coord_',rel_coord_.shape)
                inp = torch.cat([q_feat, rel_coord_], dim=-1)  # 查询点的特征表示和相对坐标拼接在一起

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1), rel_coord_.view(bs * q, -1)).view(bs, q, -1)  # 把坐标和查询特征表示一同送入MLP
                preds.append(pred)  # inp 的形状变换为 (bs * q, -1)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0];
            areas[0] = areas[3];
            areas[3] = t
            t = areas[1];
            areas[1] = areas[2];
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        # print(ret.shape)
        return ret

    def forward(self, inp, coord, cell=None):
        self.gen_feat(inp)

        # q_feat_ = self.GRN(q_feat)
        #
        # print("coord.shape", coord.shape)
        # print("coord_.shape", coord_.shape)
        return self.query_rgb(coord, cell)
