import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# ED2IF2 setting
from models.Nets import ImnetDecoder as ED2IF2_IMDecoder
from models.Nets import DeformedDecoder as ED2IF2_DeformedDecoder
from models.Nets import ResDecoder as ED2IF2_DetailDecoder
from models.pvt import pvt_tiny, pvt_small, pvt_medium, pvt_large

""" From 2D feature maps to per-point features based on pixel coordinates """
""" gt_pixels in range [0,img_res] """


def project_featmap_by_uv(uv, featmap_list):
    feat_list = []
    for featmap in featmap_list:
        feats = nn.functional.grid_sample(featmap, uv.unsqueeze(2), mode='bilinear', padding_mode="border")
        feats = feats[:, :, :, 0]
        feat_list.append(feats)
    pointfeats = torch.cat(feat_list, dim=1)
    pointfeats = pointfeats.permute(0, 2, 1)
    return pointfeats


class ED2IF2_Net(nn.Module):
    def __init__(self, config):
        super(ED2IF2_Net, self).__init__()
        # network parameters
        self.img_res = config.img_res
        self.model_dir = config.model_dir
        self.exp_name = config.exp_name
        self.iscuda = config.cuda
        self.delta = 0.01
        self.mask_weight = 0.8
        self.op_mask_weight = 0.2
        self.sdf_weight = 1.0

        # network modules
        if (config.exp_name == 'ed2if2'):
            self.image_encoder = pvt_tiny(pretrained=True)
            self.detail_decoder = ED2IF2_DetailDecoder()
            self.deformd_decoder = ED2IF2_DeformedDecoder()
            self.im_decoder = ED2IF2_IMDecoder()
        else:
            print("exp_name error!")
            exit()
        self.mseloss = nn.MSELoss()
        self.l1loss = nn.L1Loss()


        # conv kernel for computing the gradients on 2D
        kernel_nx = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        kernel_ny = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.normal_weight = torch.tensor([kernel_nx, kernel_ny]).float()
        self.normal_weight = self.normal_weight.unsqueeze(1)
        self.cam_zvec = torch.tensor(np.array([[0.0, 0.0, 1.0]])).float()
        if (self.iscuda):
            self.normal_weight = self.normal_weight.cuda()
            self.cam_zvec = self.cam_zvec.cuda()

    def get_affinities(self, points):
        pts_aff = []
        pts_aff += [torch.cat([points[:, :, 0:2], -points[:, :, 2:3]], dim=2)]  # [x,y,-z]
        return pts_aff

    """ Transform vectors from worldview to camview"""

    def project_vector_to_camview(self, vecs, transmat):
        plus = torch.zeros((vecs.size(0), vecs.size(1), 1)).cuda()
        worldvecs = torch.cat([vecs, plus], dim=2)
        camvecs = torch.matmul(worldvecs, transmat)
        return camvecs[:, :, :3]

    """ Transform vectors from camview to worldview"""

    def project_vector_to_worldview(self, vecs, transmat):
        plus = torch.tensor([[[0.0], [0.0], [0.0], [1.0]]]).cuda()
        transmat = torch.cat([transmat, plus], dim=2)
        inv_transmat = torch.inverse(transmat)
        plus = torch.zeros((vecs.size(0), vecs.size(1), 1)).cuda()
        camvecs = torch.cat([vecs, plus], dim=2)
        worldvecs = torch.matmul(camvecs, inv_transmat)
        return worldvecs[:, :, :3]

    """ Project 3D points to 2D pixels """

    def project_points_to_pixels(self, points, transmat):
        plus = torch.ones((points.size(0), points.size(1), 1)).cuda()
        homopoints = torch.cat([points, plus], dim=2)

        homopoints = torch.matmul(homopoints, transmat)
        homopoints[:, :, 0] = torch.div(homopoints[:, :, 0], homopoints[:, :, 2])
        homopoints[:, :, 1] = torch.div(homopoints[:, :, 1], homopoints[:, :, 2])

        pixels = homopoints[:, :, :2]
        uv = pixels * 2.0 / self.img_res - 1.0
        depth = homopoints[:, :, 2]

        pixels = pixels.long()
        pixels_outside = pixels < 0
        pixels = pixels.masked_fill(pixels_outside, 0)
        pixels_outside = pixels >= self.img_res
        pixels = pixels.masked_fill(pixels_outside, self.img_res - 1)
        return uv, pixels, depth

    """ From 2D feature maps to per-point features based on uv coordinates """
    """ "gt_uv" in range [-1,1] """

    def project_featmap_by_px(self, pixels, featmap):
        C = featmap.size(1)
        featmap_res = featmap.size(2)
        featmap = featmap.permute((0, 2, 3, 1))  # BCHW => BHWC
        featmap = featmap.view(featmap.size(0), -1, C)

        pointfeats = []
        for i in range(featmap.size(0)):
            pixels_per_shape = pixels[i, :, 1] * featmap_res + pixels[i, :, 0]
            point_feats_per_shape = torch.index_select(featmap[i, :, :], 0, pixels_per_shape)
            pointfeats.append(point_feats_per_shape.unsqueeze(0))
        pointfeats = torch.cat(pointfeats, dim=0)
        return pointfeats

    def forward(self, points, values, gradients, mc_image, transmat, scale):
        # image encoder
        rgba = mc_image[:, :3, :, :]
        normal = mc_image[:, 3:, :, :]
        globalfeat, local_features = self.image_encoder(rgba)

        # compute the 2D-3D correspondense
        uv, pixels, depth = self.project_points_to_pixels(points, transmat)  # BCHW => BHWC
        points_aff = self.get_affinities(points)
        uv_ref, pixels_ref, depth_ref = self.project_points_to_pixels(torch.clamp(points_aff[0], -1.0, 1.0), transmat)

        if (self.exp_name == 'ed2if2'):
            base_values = self.im_decoder(globalfeat, points)
        deformed_values, sdf1, sdf2 = self.deformd_decoder(base_values, local_features, self.img_res,
                                                                 self.img_res, uv, uv_ref, points)
        pred_disp = self.detail_decoder(local_features)

        # classify the "front-side" and "back-side" points
        projected_gradients = self.project_vector_to_camview(gradients, transmat)
        front_weights = 0.5 - torch.sign(projected_gradients[:, :, 2].unsqueeze(2)) * 0.5
        # project the per-point displacements
        pred_disp_front = pred_disp[:, 0, :, :].unsqueeze(1)
        pred_disp_back = pred_disp[:, 1, :, :].unsqueeze(1)
        pred_point_disp_front = self.project_featmap_by_px(pixels, pred_disp_front)
        pred_point_disp_back = self.project_featmap_by_px(pixels, pred_disp_back)

        # base_loss for coarse shapes
        loss_base = self.mseloss(deformed_values, values)

        # deformed_loss for deformed shapes
        loss_deformed = self.l1loss(deformed_values, values) * (3.0 / 5.0)
        loss_deformed += self.l1loss(sdf1, values) * (1.0 / 5.0)
        loss_deformed += self.l1loss(sdf2, values) * (1.0 / 5.0)

        # sdf_loss for final reconstruction
        pred_sdf_values = deformed_values + front_weights * pred_point_disp_front + (
                1 - front_weights) * pred_point_disp_back
        loss_sdf = self.l1loss(pred_sdf_values, values)

        '''
        #front
        front_values = base_values + pred_point_disp_front
        front_weights = front_weights
        loss_front = torch.sum(front_weights * torch.abs(front_values - values))/torch.sum(front_weights)
        # back
        back_values = base_values + pred_point_disp_back
        back_weights = 1-front_weights
        loss_back = torch.sum(back_weights * torch.abs(back_values - values))/torch.sum(back_weights)
        loss_sdf = (loss_front+loss_back)*0.5
        '''

        # laplacian_loss
        pred_normal_map = nn.functional.conv2d(pred_disp_front, self.normal_weight, padding=1)
        pred_nx_map = pred_normal_map[:, 0, :, :].unsqueeze(1)
        pred_ny_map = pred_normal_map[:, 1, :, :].unsqueeze(1)
        pred_nx2_map = nn.functional.conv2d(pred_nx_map, self.normal_weight, padding=1)
        pred_nx2_map = pred_nx2_map[:, 0, :, :].unsqueeze(1)
        pred_ny2_map = nn.functional.conv2d(pred_ny_map, self.normal_weight, padding=1)
        pred_ny2_map = pred_ny2_map[:, 1, :, :].unsqueeze(1)
        pred_n2_map = (pred_nx2_map + pred_ny2_map) / 2.0  # laplacian on the 2D displacement
        gt_nx_map = normal[:, 2, :, :].unsqueeze(1)
        gt_ny_map = normal[:, 1, :, :].unsqueeze(1)  # torch.Size([16, 1, 224, 224])
        gt_nx2_map = nn.functional.conv2d(gt_nx_map, self.normal_weight, padding=1)
        gt_nx2_map = gt_nx2_map[:, 0, :, :].unsqueeze(1)
        gt_ny2_map = nn.functional.conv2d(gt_ny_map, self.normal_weight, padding=1)
        gt_ny2_map = gt_ny2_map[:, 1, :, :].unsqueeze(1)
        gt_n2_map = (gt_nx2_map + gt_ny2_map) / 2.0  # gradient on the 2D normal map

        gt_Laplacian = self.project_featmap_by_px(pixels, gt_n2_map)
        pred_Laplacian = self.project_featmap_by_px(pixels, pred_n2_map)
        # Note that now the gt_Laplacian and pred_Laplacian are not matched.
        # One is "Laplacian w.r.t. 3D coordinates" and the other is "Laplacian w.r.t. 2D coordinates"
        # Next we need to project the Laplacian to 3D.

        # Note that "Laplacian computation first and then projection" is equavalent to
        # "Projection first and then Laplacian"
        # So, instead of projecting the gt_Laplacian (gt_point_n2 = gt_point_n2*(2.0*gt_depth.unsqueeze(2))/(49.0*scale)),
        # we operate on the pred_Laplacian.
        # f_u = f_v = F_MM * img_w * scale / SENSOR_SIZE_MM = (35*224*1)/(32). And note the SDF scale = 10 when loading the GT SDF in SDFdataset.py. Thus, 49.0/2.0.
        pred_Laplacian = pred_Laplacian * 49.0 * scale / (2.0 * depth.unsqueeze(2))
        lap_weights = (values < 0.1).float() * front_weights  # only consider the points near the front surface
        loss_laplacian = torch.sum(lap_weights * (pred_Laplacian + gt_Laplacian) ** 2) / torch.sum(lap_weights)
        # "+" because the displacement should be opposite to the surface geometry

        return loss_base, loss_deformed, loss_sdf, loss_laplacian

    def inference_sdf_with_detail(self, points, rgba_image, transmat):
        # compute the 3D-2D correspondence
        uv, pixels, depth = self.project_points_to_pixels(points, transmat)
        points_aff = self.get_affinities(points)
        uv_ref, pixels_ref, depth_ref = self.project_points_to_pixels(torch.clamp(points_aff[0], -1.0, 1.0), transmat)
        globalfeat, local_features = self.image_encoder(rgba_image)
        pred_disp = self.detail_decoder(local_features)

        if (self.exp_name == 'ed2if2'):
            base_values = self.im_decoder(globalfeat, points)
        deformed_values, _, _= self.deformd_decoder(base_values, local_features, self.img_res, self.img_res, uv,
                                                        uv_ref,
                                                        points)


        # compute delta
        # gradients = torch.autograd.grad(base_values, points, grad_outputs=grad_outputs, create_graph=True)
        vecs = self.cam_zvec.unsqueeze(0)
        vecs = self.project_vector_to_worldview(vecs, transmat)
        vecs = nn.functional.normalize(vecs, p=2, dim=2) * 0.03
        vecs = vecs.repeat(points.size(0), points.size(1), 1)

        # estimate the frontness
        delta_points = points + vecs
        delta_points_aff = self.get_affinities(delta_points)
        delta_uv, _, delta_depth = self.project_points_to_pixels(delta_points, transmat)
        delta_uv_ref, _, delta_depth_ref = self.project_points_to_pixels(torch.clamp(delta_points_aff[0], -1.0, 1.0), transmat)
        if (self.exp_name == 'ed2if2'):
            delta_base_values = self.im_decoder(globalfeat, delta_points)

        delta_deformed_values, _, _ = self.deformd_decoder(delta_base_values, local_features, self.img_res,
                                                              self.img_res, delta_uv, delta_uv_ref,
                                                              delta_points)
        front_weights = 0.5 - 0.5 * torch.sign(delta_deformed_values - deformed_values)

        pred_disp_front = pred_disp[:, 0, :, :].unsqueeze(1)
        pred_disp_back = pred_disp[:, 1, :, :].unsqueeze(1)
        pred_point_disp_front = project_featmap_by_uv(uv, [pred_disp_front])
        pred_point_disp_back = project_featmap_by_uv(uv, [pred_disp_back])
        final_values = delta_deformed_values + pred_point_disp_front * front_weights + pred_point_disp_back * (
                1 - front_weights)


        return final_values

