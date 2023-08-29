import os
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt
import scipy.misc
from torchvision import models

from models import ED2IF2

os.environ['TORCH_HOME'] = './ckpt/models'


#  可视化特征图
# def show_feature_map(feature_map):
#     feature_map = feature_map.squeeze(0)
#     feature_map = feature_map.cpu().numpy()
#     feature_map_num = feature_map.shape[0]
#     row_num = np.ceil(np.sqrt(feature_map_num))
#     plt.figure()
#     for index in range(1, feature_map_num + 1):
#         plt.subplot(row_num, row_num, index)
#         plt.imshow(feature_map[index - 1], cmap='gray')
#         plt.axis('off')
#         scipy.misc.imsave(str(index) + ".png", feature_map[index - 1])
    # plt.show()


def conv_relu(in_planes, out_planes, kernel_size=3, stride=1):
    # convolution and activation without changing image size
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2),
        nn.ReLU(inplace=True)
    )


def conv(in_planes, out_planes):  # Linear layer
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
    )


def sdf_init(in_dim, out_dim):  # sdf初始化
    conv_planes = [in_dim, 512, 256]
    conv1 = conv_relu(conv_planes[0], conv_planes[1], kernel_size=1)
    conv2 = conv_relu(conv_planes[1], conv_planes[2], kernel_size=1)
    convout = conv(conv_planes[2], out_dim)
    return nn.Sequential(conv1, conv2, convout)


def sdf_feature(in_dim, out_dim):
    conv_planes = [64, 128, out_dim]
    conv1 = conv_relu(in_dim, conv_planes[0], kernel_size=1)
    conv2 = conv_relu(conv_planes[0], conv_planes[1], kernel_size=1)
    convout = conv(conv_planes[1], conv_planes[2])
    return nn.Sequential(conv1, conv2, convout)


class sdf_regressor(nn.Module):
    def __init__(self, fs=1024):
        super(sdf_regressor, self).__init__()

        conv_planes = [fs, 512, 256]
        self.convs = nn.Sequential(
            conv_relu(conv_planes[0], conv_planes[1], kernel_size=1),
            conv_relu(conv_planes[1], conv_planes[2], kernel_size=1))
        self.convout = conv(conv_planes[2], 1)
        self.convout2 = conv(conv_planes[2], 256)

        self.init_weights([self.convs, self.convout, self.convout2])

    def init_weights(self, nets):
        for net in nets:
            for m in net:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_()

    def forward(self, feat):
        x = self.convs(feat)
        out1 = self.convout(x)
        out2 = self.convout2(x)
        return out1, out2


class ResEncoder(nn.Module):
    def __init__(self, res_dir):
        super(ResEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.fc1 = nn.Linear(1000, 128)

    def forward(self, input_view):
        feat0 = self.relu(self.bn1(self.conv1(input_view)))  # torch.Size([B, 64, 224, 224])
        x = self.maxpool(feat0)

        feat1 = self.layer1(x)  # torch.Size([B, 64, 112, 112])
        feat2 = self.layer2(feat1)  # torch.Size([B, 128, 56, 56])
        feat3 = self.layer3(feat2)  # torch.Size([B, 256, 28, 28])
        feat4 = self.layer4(feat3)  # torch.Size([B, 512, 14, 14])

        x = self.avgpool(feat4)  # torch.Size([B, 512, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([B, 512])
        featvec = self.fc(x)  # torch.Size([B, 1000])
        featvec = self.fc1(featvec)  # torch.Size([B, 128])
        # print(featvec.shape)
        # all_featmaps = [feat0, feat1, feat2, feat3, feat4, feat5]
        featmap_list = [feat0, feat1, feat2, feat3, feat4]
        return featvec, featmap_list
        # return featmap_list0, featmap_list
        # return all_featmaps, mid_featmaps


class ImnetDecoder(nn.Module):
    def __init__(self):
        super(ImnetDecoder, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32
        # self.fc = nn.Linear(512, 1000) # B*C
        # self.fc1 = nn.Linear(1000, self.global_feat_dim)

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim + 3, self.latent_feat_dim * 16)
        self.l2 = nn.Linear(self.latent_feat_dim * 16 + 3, self.latent_feat_dim * 8)
        self.l3 = nn.Linear(self.latent_feat_dim * 8 + 3, self.latent_feat_dim * 4)
        self.l4 = nn.Linear(self.latent_feat_dim * 4 + 3, self.latent_feat_dim * 2)
        self.l5 = nn.Linear(self.latent_feat_dim * 2 + 3, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

    #     # self.sdf_template = sdf_init(4, 1)
    #     self.sdffeat = sdf_feature(1, 256)
    #     self.cls_local_3 = sdf_regressor(512 * 2 + 256 + 1)  # 1792
    #     self.cls_local_2 = sdf_regressor((512 + 320) * 2 + 512 + 1)
    #     self.cls_local_1 = sdf_regressor((320 + 128) * 2 + 512 + 1)
    #     self.cls_local_0 = sdf_regressor((128 + 64) * 2 + 512 + 1)
    #     # self.cls_local_1 = sdf_regressor((128 + 64) * 2 + 512 + 1)
    #     # self.cls_local_ = sdf_regressor((64 + 64) * 2 + 3 + 512)
    #
    #     self.init_weights([self.sdffeat])
    #
    # def init_weights(self, nets):
    #     for net in nets:
    #         for m in net:
    #             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #                 nn.init.xavier_uniform_(m.weight.data)
    #                 if m.bias is not None:
    #                     m.bias.data.zero_()
    #             if isinstance(m, nn.Linear):
    #                 m.weight.data.normal_()

    # def forward(self, featmaps, img_h, img_w, uv, uv_ref, points, depth):
    def forward(self, globalfeat, points):
        """get_local_image_features"""
        # print(pixs.shape) torch.Size([16, 2048, 2])
        # [feat0, feat1, feat2, feat3, feat4] = featmaps
        # feat4 = torch.flatten(feat4, 1)
        # feat4 = self.fc(feat4)
        # feat4 = self.fc1(feat4)
        # feature = feat4.unsqueeze(1)
        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1, points.size(1), 1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l2(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l3(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l4(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l5(feature))
        feature = self.relu(self.l6(feature))
        feature = self.l7(feature)

        base_sdf = torch.reshape(feature, (points.size(0), -1, 1))
        return base_sdf

        # pred_sdf4 = pred_sdf5.unsqueeze(1)
        #
        # fms = []
        # fms += [F.interpolate(feat0, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 64, 224, 224])，特征图通道数量不变，尺寸变为指定size
        # fms += [F.interpolate(feat1, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 128, 224, 224])
        # fms += [F.interpolate(feat2, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 320, 224, 224])
        # fms += [F.interpolate(feat3, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 512, 224, 224])
        # fms += [F.interpolate(feat4, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 512, 224, 224])

        # local_feats = []
        # local_feats_ref = []
        # s = float(img_h) // 2
        # if len(pixs) >= 1:
        #     t = []
        #     for px in pixs:
        #         # print(px.shape)
        #         t += [((px - s) / s).unsqueeze(2)] # 计算uv坐标
        #     pixs = t
        #     # print(pixs)
        #
        # if len(pixs_ref) >= 1:
        #     t = []
        #     for px in pixs_ref:
        #         t += [((px - s) / s).unsqueeze(2)]
        #     pixs_ref = t

        # local_feats = D2IM.project_featmap_by_uv(uv, fms)
        # local_feats_ref = D2IM.project_featmap_by_uv(uv_ref, fms)
        # feat_list = []
        # feat_list_ref = []
        # for x in fms:
        # point sampler;
        # print(x.shape)
        # print(uv.shape)
        # feats = F.grid_sample(x, uv.unsqueeze(2), mode='bilinear', align_corners=True)
        # feats = feats[:, :, :, 0]
        # print(feats.shape)
        # print("-----------------------------------------------------------------")
        # local_feats.append(feats)
        # feats_ref = F.grid_sample(x, uv_ref.unsqueeze(2), mode='bilinear', align_corners=True)
        # local_feats_ref.append(feats_ref)
        # if len(pixs) >= 1:
        #     x1 = []
        #     for px in pixs:
        #         x1.append([F.grid_sample(x, px, mode='bilinear', align_corners=True)])
        #     local_feats += [torch.cat(x1, dim=1)]  # localfeats

        # if len(pixs_ref) >= 1:
        #     x_ref = []
        #     for px in pixs_ref:
        #         x_ref.append([F.grid_sample(x, px, mode='bilinear', align_corners=True)])
        #     local_feats_ref += [torch.cat(x_ref, dim=1)]  # localfeats_ref

        # points_transposed_unsqueezed = points.transpose(2, 1).unsqueeze(3)
        # depth_transposed_unsqueezed = depth.unsqueeze(1).unsqueeze(3)
        # print(depth_transposed_unsqueezed.shape)
        # print(points_transposed_unsqueezed.shape)
        # pred_sdf5 = self.sdf_template(torch.cat([points_transposed_unsqueezed, depth_transposed_unsqueezed],dim=1))  # B * 3 * N * 1 => B * 1 * N * 1

        # 第一个变形SDF模块
        # print(local_feats[5].shape)
        # print(local_feats_ref[5].shape)
        # lf5 = torch.cat([local_feats[5], local_feats_ref[5]], dim=1)
        # print(lf5.shape)
        # 第一个变形SDF模块
        # lf3 = torch.cat([local_feats[3], local_feats_ref[3]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf4)  # B * 256 * N * 1
        # # print(sdffeat.shape)
        # delta_sdf4, statevec = self.cls_local_3(torch.cat([lf3, sdffeat, depth_transposed_unsqueezed], dim=1))
        # pred_sdf3 = delta_sdf4 + pred_sdf4
        #
        # # 第二个变形SDF模块
        # # 第二个变形SDF模块
        # lf2 = torch.cat([local_feats[3], local_feats_ref[3], local_feats[2], local_feats_ref[2]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf3)  # B * C * N * 1
        # delta_sdf3, statevec = self.cls_local_2(
        #     torch.cat([lf2, sdffeat, statevec, depth_transposed_unsqueezed], dim=1))
        # pred_sdf2 = delta_sdf3 + pred_sdf3
        #
        # # 第三个变形SDF模块
        # # 第三个变形SDF模块
        # lf1 = torch.cat([local_feats[2], local_feats_ref[2], local_feats[1], local_feats_ref[1]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf2)  # B * C * N * 1
        # delta_sdf2, statevec = self.cls_local_1(
        #     torch.cat([lf1, sdffeat, statevec, depth_transposed_unsqueezed], dim=1))
        # pred_sdf1 = delta_sdf2 + pred_sdf2
        #
        # # 第四个变形SDF模块
        # # 第四个变形SDF模块
        # lf0 = torch.cat([local_feats[1], local_feats_ref[1], local_feats[0], local_feats_ref[0]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf2)  # B * C * N * 1
        # delta_sdf1, statevec = self.cls_local_0(
        #     torch.cat([lf0, sdffeat, statevec, depth_transposed_unsqueezed], dim=1))
        # base_sdf = (delta_sdf1 + pred_sdf1).squeeze(1)
        #
        # # 第五个变形SDF模块
        # # lf0 = torch.cat([local_feats[1], local_feats_ref[1], local_feats[0], local_feats_ref[0]], dim=1)
        # # sdffeat = self.sdffeat(pred_sdf1)  # B * C * N * 1
        # # delta_sdf0, statevec = self.cls_local_1(
        # #     torch.cat([lf0, points_transposed_unsqueezed, sdffeat, statevec, depth_transposed_unsqueezed], dim=1))
        # # base_sdf = (delta_sdf0 + pred_sdf1).squeeze(1)
        #
        # # 第六个变形SDF模块
        # # lf0 = torch.cat([local_feats[1], local_feats_ref[1], local_feats[0], local_feats_ref[0]], dim=1)
        # # sdffeat = self.sdffeat(pred_sdf1)  # B * C * N * 1
        # # delta_sdf, statevec = self.cls_local_1(torch.cat([lf0, points_transposed_unsqueezed, sdffeat, statevec], dim=1))
        # # base_sdf = (delta_sdf + pred_sdf1).squeeze(1)
        #
        # return base_sdf


class DeformedDecoder(nn.Module):
    def __init__(self):
        super(DeformedDecoder, self).__init__()

        self.sdffeat = sdf_feature(1, 256)
        # self.sdf_template = sdf_init(3, 1)
        # self.cls_local_3 = sdf_regressor(512 * 2 + 256 + 3)  # 1792
        # self.cls_local_3 = sdf_regressor((512 + 320) * 2 + 256 + 3)
        self.cls_local_2 = sdf_regressor((512 + 320) * 2 + 256 + 3)
        self.cls_local_1 = sdf_regressor((320 + 128) * 2 + 512 + 3)
        self.cls_local_0 = sdf_regressor((128 + 64) * 2 + 512 + 3)

        self.init_weights([self.sdffeat])

    def init_weights(self, nets):
        for net in nets:
            for m in net:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_()

    # def forward(self, featmap_list, img_h, img_w, uv, uv_ref, points):
    def forward(self, base_sdf, featmap_list, img_h, img_w, uv, uv_ref, points):
        pred_sdf3 = base_sdf.unsqueeze(1)
        [feat0, feat1, feat2, feat3] = featmap_list
        # ablation
        fms = []
        # fms += [F.interpolate(feat0, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 64, 224, 224])，特征图通道数量不变，尺寸变为指定size
        # fms += [F.interpolate(feat1, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 64, 224, 224])
        # fms += [F.interpolate(feat2, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 128, 224, 224])
        # fms += [F.interpolate(feat3, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 256, 224, 224])
        # fms += [F.interpolate(feat4, size=[img_h, img_w], mode='bilinear',
        #                       align_corners=False)]  # torch.Size([B, 512, 224, 224])

        fms += [F.interpolate(feat0, size=[img_h, img_w], mode='bilinear',
                              align_corners=False)]  # torch.Size([B, 64, 224, 224])，特征图通道数量不变，尺寸变为指定size
        fms += [F.interpolate(feat1, size=[img_h, img_w], mode='bilinear',
                              align_corners=False)]  # torch.Size([B, 128, 224, 224])
        fms += [F.interpolate(feat2, size=[img_h, img_w], mode='bilinear',
                              align_corners=False)]  # torch.Size([B, 320, 224, 224])
        fms += [F.interpolate(feat3, size=[img_h, img_w], mode='bilinear',
                              align_corners=False)]  # torch.Size([B, 512, 224, 224])

        local_feats = []
        local_feats_ref = []
        for x in fms:
            feats = F.grid_sample(x, uv.unsqueeze(2), mode='bilinear', align_corners=True)
            local_feats.append(feats)
            feats_ref = F.grid_sample(x, uv_ref.unsqueeze(2), mode='bilinear', align_corners=True)
            local_feats_ref.append(feats_ref)

        points_transposed_unsqueezed = points.transpose(2, 1).unsqueeze(3)
        # pred_sdf3 = self.sdf_template(points_transposed_unsqueezed)
        # 第一个变形SDF模块
        # lf3 = torch.cat([local_feats[3], local_feats_ref[3]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf4)  # B * 256 * N * 1
        # delta_sdf4, statevec = self.cls_local_3(torch.cat([lf3, sdffeat, points_transposed_unsqueezed], dim=1))
        # pred_sdf3 = delta_sdf4 + pred_sdf4

        # 第一个变形SDF模块
        # lf3 = torch.cat([local_feats[4], local_feats_ref[4], local_feats[3], local_feats_ref[3]], dim=1)
        # sdffeat = self.sdffeat(pred_sdf4)  # B * C * N * 1
        # delta_sdf4, statevec = self.cls_local_3(
        #     torch.cat([lf3, sdffeat, points_transposed_unsqueezed], dim=1))
        # pred_sdf3 = delta_sdf4 + pred_sdf4

        # 第二个变形SDF模块
        lf2 = torch.cat([local_feats[3], local_feats_ref[3], local_feats[2], local_feats_ref[2]], dim=1)
        sdffeat = self.sdffeat(pred_sdf3)  # B * C * N * 1
        delta_sdf3, statevec = self.cls_local_2(
            torch.cat([lf2, sdffeat, points_transposed_unsqueezed], dim=1))
        pred_sdf2 = delta_sdf3 + pred_sdf3

        # 第三个变形SDF模块
        lf1 = torch.cat([local_feats[2], local_feats_ref[2], local_feats[1], local_feats_ref[1]], dim=1)
        sdffeat = self.sdffeat(pred_sdf2)  # B * C * N * 1
        delta_sdf2, statevec = self.cls_local_1(
            torch.cat([lf1, sdffeat, statevec, points_transposed_unsqueezed], dim=1))
        # deformed_sdf = (delta_sdf1 + pred_sdf1).squeeze(1)
        pred_sdf1 = delta_sdf2 + pred_sdf2

        # 第四个变形SDF模块
        lf0 = torch.cat([local_feats[1], local_feats_ref[1], local_feats[0], local_feats_ref[0]], dim=1)
        sdffeat = self.sdffeat(pred_sdf1)  # B * C * N * 1
        delta_sdf1, statevec = self.cls_local_0(
            torch.cat([lf0, sdffeat, statevec, points_transposed_unsqueezed], dim=1))
        deformed_sdf = delta_sdf1 + pred_sdf1

        # return deformed_sdf
        return deformed_sdf.squeeze(1), pred_sdf1.squeeze(1), pred_sdf2.squeeze(1)


class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)  # torch.Size([B, C, 1, 1])
        F_max = self.max_pool(x)  # torch.Size([B, C, 1, 1])
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max  # torch.Size([B, C, 1, 1])
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)  # torch.Size([B, 1, C])
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)  # torch.Size([B, C, 1, 1])
        out = self.sigmoid(F_add_)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    def get_im_subim_channels(self, C_im, M):
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature


class ResBlock_HAM(nn.Module):
    def __init__(self, Channel_nums):
        super(ResBlock_HAM, self).__init__()
        self.channel = Channel_nums
        self.ChannelAttention = ChannelAttention(self.channel)
        self.SpatialAttention = SpatialAttention(self.channel)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        residual = x_in
        channel_attention_map = self.ChannelAttention(x_in)
        channel_refined_feature = channel_attention_map * x_in
        final_refined_feature = self.SpatialAttention(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return out


class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel = 64

        # self.up_conv5 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        self.up_conv4 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        self.up_conv3 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        self.up_conv2 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        self.up_conv1 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        self.up_conv0 = nn.Conv2d(self.channel, self.channel, (1, 1))  # yes
        # self.up_conv0 = nn.Conv2d(self.channel, self.channel, (1, 1))

        self.resham4 = ResBlock_HAM(512)  # yes
        self.resham3 = ResBlock_HAM(320)  # yes
        self.resham2 = ResBlock_HAM(128)  # yes
        self.resham1 = ResBlock_HAM(64)  # yes
        self.resham0 = ResBlock_HAM(self.channel)  # yes

        # self.c5_conv = nn.Conv2d(512, self.channel, (1, 1))  # yes
        self.c4_conv = nn.Conv2d(512, self.channel, (1, 1))  # yes
        self.c3_conv = nn.Conv2d(320, self.channel, (1, 1))  # yes
        self.c2_conv = nn.Conv2d(128, self.channel, (1, 1))  # yes
        self.c1_conv = nn.Conv2d(64, self.channel, (1, 1))  # yes
        # self.c1_conv = nn.Conv2d(64, self.channel, (1, 1))

        self.p0_conv = nn.Conv2d(self.channel, self.channel, (3, 3), padding=1)
        self.pred_disp = nn.Conv2d(self.channel, 2, (1, 1), padding=0)
        self.relu = nn.ReLU()
        # nn.init.xavier_uniform_(self.up_conv5.weight.data)
        # self.up_conv5.bias.data.zero_()
        nn.init.xavier_uniform_(self.up_conv4.weight.data)
        self.up_conv4.bias.data.zero_()
        nn.init.xavier_uniform_(self.up_conv3.weight.data)
        self.up_conv3.bias.data.zero_()
        nn.init.xavier_uniform_(self.up_conv2.weight.data)
        self.up_conv2.bias.data.zero_()
        nn.init.xavier_uniform_(self.up_conv1.weight.data)
        self.up_conv1.bias.data.zero_()
        nn.init.xavier_uniform_(self.up_conv0.weight.data)
        self.up_conv0.bias.data.zero_()
        # nn.init.xavier_uniform_(self.c5_conv.weight.data)
        # self.c5_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.c4_conv.weight.data)
        self.c4_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.c3_conv.weight.data)
        self.c3_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.c2_conv.weight.data)
        self.c2_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.c1_conv.weight.data)
        self.c1_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.p0_conv.weight.data)
        self.p0_conv.bias.data.zero_()
        nn.init.xavier_uniform_(self.pred_disp.weight.data)
        self.pred_disp.bias.data.zero_()


    def forward(self, featmap_list):
        # ablation study
        # [feat0, feat1, feat2, feat3, feat4] = featmap_list
        # p5 = self.relu(self.c5_conv(self.resham4(feat4)))
        # p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(self.resham3(feat3)))
        # p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(self.resham2(feat2)))
        # p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(self.resham1(feat1)))
        # p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(self.resham0(feat0)))
        # p0 = self.relu(self.p0_conv(self.resham0(p1)))
        # output_disp = self.pred_disp(p0)
        # return output_disp
        [feat0, feat1, feat2, feat3] = featmap_list
        # print(feat3.shape)
        # print(feat2.shape)
        # print(feat1.shape)
        # print(feat0.shape)
        p4 = self.relu(self.c4_conv(self.resham4(feat3)))
        # print(p4.shape)
        # print(self.up_conv4(self.upsample(p4)).shape)
        # print(self.relu(self.c3_conv(self.resham3(feat2))).shape)
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(self.resham3(feat2)))

        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(self.resham2(feat1)))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(self.resham1(feat0)))
        p1 = self.relu(self.up_conv0(self.upsample(self.relu(self.up_conv1(self.upsample(p1))))))
        p0 = self.relu(self.p0_conv(self.resham0(p1)))
        output_disp = self.pred_disp(p0)
        # show_feature_map(output_disp)
        return output_disp

# class ResDecoder(nn.Module):
#     def __init__(self):
#         super(ResDecoder, self).__init__()
#         self.channel = 64  # 目标维度
#         self.sin_conv2d = Sin_Conv(512, self.channel)
#         self.up_linear1 = Upsam_Linear(320, self.channel)
#         self.up_linear2 = Upsam_Linear(128, self.channel)
#         self.up_linear3 = Upsam_Linear(64, self.channel)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.conv = nn.Sequential(nn.Conv2d(self.channel, self.channel, (1,1)))
#         # self.up_linear4 = Upsam_Linear(64, self.channel)
#         self.sin_conv = Sin_Convs()
#         # self.p0_conv = nn.Sequential(nn.Conv2d(self.channel, self.channel, (3, 3), padding=1))
#         self.pred_disp = nn.Sequential(nn.Conv2d(self.channel, 2, (1, 1), padding=0))
#         self.first_layer_init = first_layer_sine_init
#         self.init_weights(
#             [self.sin_conv2d, nn.Sequential(self.up_linear1, self.up_linear2, self.up_linear3),
#              self.sin_conv, self.conv, self.pred_disp])
#         self.sin_conv2d.apply(self.first_layer_init)
#
#         """
#         self.up_conv5 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#         self.up_conv4 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#         self.up_conv3 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#         self.up_conv2 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#         self.up_conv1 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#         self.up_conv0 = nn.Linear(self.channel, self.channel)  # 用于提取深度特征，不改变特征通道数量以及尺寸
#
#         self.c5_conv = nn.Linear(512, self.channel)  # 用于降低维度
#         self.c4_conv = nn.Linear(256, self.channel)  # 用于降低维度
#         self.c3_conv = nn.Linear(128, self.channel)  # 用于降低维度
#         self.c2_conv = nn.Linear(64, self.channel)  # 用于降低维度
#         self.c1_conv = nn.Linear(64, self.channel)  # 用于降低维度
#
#         self.p0_conv = nn.Conv2d(self.channel, self.channel, (3, 3), padding=1)
#         self.pred_disp = nn.Linear(self.channel, 2)
#         self.relu = nn.ReLU()
#         """
#
#     def init_weights(self, nets):
#         for net in nets:
#             for m in net:
#                 sine_init(m)
#
#     def forward(self, featmap_list):
#         """get displacement maps"""
#         [feat0, feat1, feat2, feat3] = featmap_list
#         p4 = self.sin_conv2d(feat3)
#         p3 = self.up_linear1(p4, feat2)
#         p2 = self.up_linear2(p3, feat1)
#         p1 = self.up_linear3(p2, feat0)
#         # p1 = self.up_linear3(p2, feat0)
#         p0 = self.sin_conv(p1)
#         p0 = self.conv(self.upsample(self.upsample(self.upsample(p0))))
#         output_disp = self.pred_disp(p0)
#         return output_disp
#
#
# def Sin_Conv(in_planes, out_planes):
#     return nn.Sequential(
#         nn.Conv2d(in_planes, out_planes, (1, 1)),
#         Sine()
#     )
#
#
# class Upsam_Linear(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(Upsam_Linear, self).__init__()
#         self.channel = 64
#         self.conv = nn.Conv2d(self.channel, self.channel, (1, 1))
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.sin_conv = Sin_Conv(in_planes, out_planes)
#
#     def forward(self, input0, input1):
#         output = self.conv(self.upsample(input0)) + self.sin_conv(input1)
#         return output
#
#
# def Sin_Convs(channel=64):
#     return nn.Sequential(
#         nn.Conv2d(channel, channel, (3, 3), padding=1),
#         Sine()
#     )
#
#
#
# class Sine(nn.Module):
#     def __init(self):
#         super().__init__()
#
#     def forward(self, input):
#         # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
#         return torch.sin(30 * input)
#
#
# def sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             # See supplement Sec. 1.5 for discussion of factor 30
#             m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
#
#
# def first_layer_sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
#             m.weight.uniform_(-1 / num_input, 1 / num_input)


# class Linear(nn.Module):
#     def __init(self, in_features, out_features):
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.Linear = nn.Linear(self.in_features, self.out_features)
#
#     def forward(self, feature, points):
#         feature = torch.cat([feature, points], dim=2)
#         feature = self.Linear(feature)
#         return feature


# if __name__ == "__main__":
#     ImnetDecoder = ImnetDecoder()
#     ImnetDecoder([torch.rand([4, 64, 224, 224]), torch.rand(4, 64, 112, 112), torch.rand(4, 128, 56, 56),
#                 torch.rand(4, 256, 28, 28), torch.rand(4, 512, 14, 14), torch.rand(4, 512, 1, 1)], 224, 224)

# if __name__ == "__main__":
#     ResEncoder = ResEncoder("./ckpt/models'")
#     input_view = torch.rand([4, 3, 224, 224])
#     featmap_list0, featmap_list = ResEncoder(input_view)
#     print(len(featmap_list0))
#     print(len(featmap_list))
