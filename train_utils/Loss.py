from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .dice_coefficient_loss import dice_loss, cal_sp_dice
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from .skeletonize import Skel_type


def criterion(output, target, alpha, beta, num_classes, half_size, ignore_index, skel_type, num_iter=5):
    target = build_target(target, num_classes, ignore_index)
    # el = EulerLayer(half_size).to(output.device)
    skel = Skel_type(num_iter, output.device, num_classes, half_size, skel_type)

    if num_classes == 1:
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]).to(output.device))
        bceloss = bce(output, target)
        # diceloss = dice_loss(torch.sigmoid(output), target, multiclass=False, ignore_index=ignore_index)
        skelloss = cal_sp_dice(torch.sigmoid(output), skel(torch.sigmoid(output)), target, skel(target), ignore_index)

    else:
        bce = nn.CrossEntropyLoss(weight=torch.as_tensor([1.0, 1.0], device=output.device))
        bceloss = bce(output, target)
        # diceloss = dice_loss(F.softmax(output, dim=1), target, multiclass=True, ignore_index=ignore_index)
        skelloss = cal_sp_dice(torch.softmax(output, dim=1), skel(torch.softmax(output, dim=1)), target, skel(target),
                               ignore_index)

    if alpha == 0.0:
        loss = bceloss
    else:
        loss = bceloss + alpha * skelloss

    return loss


def criterion_ssp(output, v, target, alpha, beta, num_classes, half_size, ignore_index, skel_type, num_iter=5):
    target = build_target(target, num_classes, ignore_index)
    skel = Skel_type(num_iter, output.device, num_classes, half_size, skel_type)
    # l1 = nn.SmoothL1Loss()

    if num_classes == 1:
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1]).to(output.device))
        bceloss = bce(output, target)
        # bceloss_v = bce(v, target)
        # l1loss = l1(skel(torch.sigmoid(v)), skel(target))
        skelloss = cal_sp_dice(torch.sigmoid(v), skel(torch.sigmoid(v)), target, skel(target), ignore_index)
        # skelloss = cal_sp_dice(torch.sigmoid(output), skel(torch.sigmoid(output)), target, skel(target), ignore_index)

    else:
        bce = nn.CrossEntropyLoss(weight=torch.as_tensor([1.0, 1.0], device=output.device))
        bceloss = bce(output, target)
        skelloss = cal_sp_dice(torch.softmax(output, dim=1), skel(torch.softmax(output, dim=1)), target, skel(target),
                               ignore_index)

    if alpha == 0.0:
        loss = bceloss
    else:
        loss = bceloss + alpha * skelloss

    return loss


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)

        dice_target[ignore_mask] = 0

        if num_classes == 1:
            dice_target = dice_target.unsqueeze(-1).float()
            dice_target[ignore_mask] = ignore_index
        else:
            # [N, H, W] -> [N, H, W, C]
            dice_target = nn.functional.one_hot(dice_target, num_classes).float()
            dice_target[ignore_mask] = ignore_index
    else:
        if num_classes == 1:
            dice_target = dice_target.unsqueeze(-1).float()
        else:
            dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def grad_f(img):
    b, num_classes, h, w = img.shape
    nabla = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]],
                          [[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], requires_grad=False).to(img.device)
    nabla = F.conv2d(img, weight=nabla.repeat(num_classes, 1, 1, 1), stride=1, padding=1, groups=num_classes)
    # grads_reshaped = nabla.view(b, num_classes, 2, h, w)
    #
    # # 计算二范数 (b, num_classes, 1, h, w)
    # norm = torch.norm(grads_reshaped, p=2, dim=2, keepdim=True)
    # norm = torch.clamp(norm, min=1e-8)
    # # 归一化
    # normalized_grads = grads_reshaped / norm
    # return normalized_grads.view(b, -1, h, w)

    return nabla


def div_f(img, num_classes):
    div = torch.tensor([[[[1., 0., -1.],
                          [2., 0., -2.],
                          [1., 0., -1.]],
                         [[1., 2., 1.],
                          [0., 0., 0.],
                          [-1., -2., -1.]]]], requires_grad=False).to(img.device)
    div = F.conv2d(img, weight=div.repeat(num_classes, 1, 1, 1), stride=1, padding=1, groups=num_classes)
    return div


def shape_field(grad_1, grad_2):
    # dim = 2
    g1, g2 = grad_1.chunk(2, dim=1)
    u1, u2 = grad_2.chunk(2, dim=1)
    cross_up = u1 * g2 - u2 * g1
    p = torch.cat([-cross_up * g2, cross_up * g1], dim=1)
    return div_f(p, num_classes=1)


def Vis_Field(vector_field_np, filename):
    height, width = vector_field_np.shape[0:2]
    sample_step = 8
    Y, X = np.meshgrid(np.arange(height), np.arange(width))
    dpi = 100  # 假设 DPI = 100
    figsize = (width / dpi, height / dpi)

    plt.figure(figsize=figsize)
    plt.quiver(Y[::sample_step, ::sample_step], X[::sample_step, ::sample_step],
               vector_field_np[::sample_step, ::sample_step, 0], vector_field_np[::sample_step, ::sample_step, 1],
               angles='xy',
               scale_units='xy', scale=0.1, color='b')

    plt.gca().invert_yaxis()
    plt.axis('off')
    # 保存图片
    plt.savefig(f"./field/{filename}", dpi=dpi, bbox_inches='tight', pad_inches=0)

    # 关闭 figure，避免显示
    plt.close()


def DT(img, iter_=50):
    # trans = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    #
    # dt_map = img
    # out_trans = trans(dt_map)
    # in_trans = -trans(-dt_map)
    # for i in range(iter_):
    #     dt_map = dt_map + out_trans + in_trans
    #     out_trans = trans(out_trans)
    #     in_trans = -trans(-in_trans)

    """
    使用 scipy 的距离变换（需要转到 CPU 和 numpy）
    """
    binary_mask = (img >= 0.5).float()
    if binary_mask.is_cuda:
        binary_mask_np = binary_mask.cpu().numpy()
    else:
        binary_mask_np = binary_mask.numpy()

    distance_np = distance_transform_edt(binary_mask_np) + distance_transform_edt(1 - binary_mask_np)
    dt_map = torch.from_numpy(distance_np).to(binary_mask.device).float()

    return dt_map


class STDLayer(nn.Module):

    def __init__(
            self,
            nb_classes,
            device,
            nb_iterations=10,
            nb_kerhalfsize=7,
            l_ker=False
    ):
        """
        :param nb_classes: number of classes
        :param nb_iterations: iterations number
        :param nb_kerhalfsize: the half size of neigbourhood
        """
        super(STDLayer, self).__init__()

        self.device = device
        self.nb_iterations = nb_iterations
        self.nb_classes = nb_classes
        self.ker_halfsize = nb_kerhalfsize
        self.l_ker = l_ker

        if l_ker:
            self.nbsigma = nn.Parameter(torch.FloatTensor([1.0] * nb_classes).view(nb_classes, 1, 1),
                                        requires_grad=True)
            self.conv2d = nn.Conv2d(nb_classes, nb_classes, kernel_size=1 + 2 * nb_kerhalfsize, padding=nb_kerhalfsize,
                                    groups=nb_classes)
        else:
            self.nbsigma = nn.Parameter(torch.FloatTensor([1.0] * nb_classes).view(nb_classes, 1, 1),
                                        requires_grad=True)

        self.lam = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.entropy_epsilon = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # softmax
        self.softmax = nn.Softmax2d()

    def STD_Kernel(self, sigma, halfsize):
        x, y = torch.meshgrid(torch.arange(-halfsize, halfsize + 1), torch.arange(-halfsize, halfsize + 1))
        x = x.to(self.device)
        y = y.to(self.device)
        ker = torch.exp(-(x.float() ** 2 + y.float() ** 2) / (2.0 * (sigma ** 2)))
        ker = ker / (ker.sum(-1, keepdim=True).sum(-2, keepdim=True) + 1e-15)
        ker = ker.unsqueeze(1)
        return ker

    def p(self, gt):
        ker = self.STD_Kernel(self.nbsigma, self.ker_halfsize)

        p = F.conv2d(1.0 - 2.0 * gt, ker, padding=self.ker_halfsize, groups=self.nb_classes)

        return p

    def forward(self, o):

        # std kernel
        ker = self.STD_Kernel(self.nbsigma, self.ker_halfsize)

        # main iteration
        for i in range(self.nb_iterations):
            if i == 0:
                u = self.softmax(o * (self.entropy_epsilon ** 2))
            else:
                u = self.softmax((o - self.lam * p) * (self.entropy_epsilon ** 2))

            # 1. subgradient
            if self.l_ker:
                p = self.conv2d(F.conv2d(1.0 - 2.0 * u, ker, padding=self.ker_halfsize, groups=self.nb_classes))

            else:
                p = F.conv2d(1.0 - 2.0 * u, ker, padding=self.ker_halfsize, groups=self.nb_classes)

        o_new = (o - self.lam * p) * (self.entropy_epsilon ** 2)

        return o_new


def connected_components(image: Tensor, conn_type: int, num_iterations: int = 100) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError(f"Input imagetype is not a Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    bs, c, h, w = image.shape

    mask = torch.arange(1, h * w + 1, device=image.device, dtype=image.dtype).view((1, 1, h, w)).expand((bs, c, h, w))
    mask = torch.mul(mask, image)

    # morph = MorphLayer(image.device, c, half_kersize=1)

    for _ in range(num_iterations):
        if conn_type == 8:
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        #
        elif conn_type == 4:
            mask1 = F.max_pool2d(mask, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
            mask2 = F.max_pool2d(mask, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
            mask = torch.max(mask1, mask2)
        mask = torch.mul(mask, image)  # mask using element-wise multiplication

        # mask = morph.dilate(mask, is_soft=True, conn_type=conn_type)

    return mask.view_as(image)


def patchify(image, half_size=1):
    B, C, H, W = image.shape
    ker_size = 2 * half_size + 1
    image_patch = nn.Unfold(kernel_size=(ker_size, ker_size), padding=1, stride=half_size)(image.float())
    # divide the image into patches via sliding window
    # image patch, shape [B,C*9,H*W]
    image_patch = image_patch.permute(0, 2, 1).reshape(B * H * W, C, ker_size, ker_size)  # B, H*W, C*9
    # print(image_patch.shape)

    return image_patch


def to_po_sittion_loss(output, target, half_size=1):
    output_patch_fore = patchify(output[:, 1, ...].unsqueeze(1), half_size)
    target_patch_fore = patchify(target[:, 1, ...].unsqueeze(1), half_size)
    output_patch_back = patchify(output[:, 1, ...].unsqueeze(1), half_size)
    target_patch_back = patchify(target[:, 1, ...].unsqueeze(1), half_size)

    iter_ = 2 * half_size + 1

    topo_out4 = connected_components(output_patch_fore, 4, num_iterations=3 * iter_)
    topo_gt4 = connected_components(target_patch_fore, 4, num_iterations=3 * iter_)

    topo_out8 = connected_components(output_patch_back, 8, num_iterations=2 * iter_)
    topo_gt8 = connected_components(target_patch_back, 8, num_iterations=2 * iter_)

    L = nn.MSELoss()
    # L = nn.L1Loss()
    topoloss = L(topo_gt8, topo_out8) + L(topo_gt4, topo_out4)

    return topoloss


class EulerLayer(nn.Module):
    def __init__(self, area_size):
        super(EulerLayer, self).__init__()

        self.k1 = nn.Parameter(torch.tensor([[[[0., 1., 0.], [1., 1., 0.], [0., 0., 0.]]]], requires_grad=False))
        self.k2 = nn.Parameter(torch.tensor([[[[0., 1., 0.], [0., 1., 1.], [0., 0., 0.]]]], requires_grad=False))
        self.k3 = nn.Parameter(torch.tensor([[[[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]]]], requires_grad=False))
        self.k4 = nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., 1., 1.], [0., 1., 0.]]]], requires_grad=False))

        self.avgpool_point_pad = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=(1, 1),
                                              count_include_pad=True, divisor_override=1)
        self.avgpool_point = nn.AvgPool2d(kernel_size=(2, 2), stride=1, padding=(0, 0),
                                          count_include_pad=True, divisor_override=1)

        self.avgpool_row_pad = nn.AvgPool2d(kernel_size=(2, 1), stride=1, padding=(1, 0),
                                            count_include_pad=True, divisor_override=1)

        self.avgpool_row = nn.AvgPool2d(kernel_size=(2, 1), stride=1, padding=(0, 0),
                                        count_include_pad=True, divisor_override=1)

        self.avgpool_col_pad = nn.AvgPool2d(kernel_size=(1, 2), stride=1, padding=(0, 1),
                                            count_include_pad=True, divisor_override=1)
        self.avgpool_col = nn.AvgPool2d(kernel_size=(1, 2), stride=1, padding=(0, 0),
                                        count_include_pad=True, divisor_override=1)

        self.mse = nn.MSELoss()
        self.klloss = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()
        self.huber = nn.HuberLoss(delta=1)

        self.avgpool_area = nn.AvgPool2d(kernel_size=area_size, count_include_pad=True, divisor_override=1)

    def euler_point(self, img, conn_type):
        if conn_type == 8:
            point = self.avgpool_point_pad(img)
            # point = torch.where(point > 0, 1.0 / point, torch.zeros_like(point))
            point = torch.where(point != 0, 1.0 / point, torch.zeros_like(point))
            point = self.avgpool_point(point)
            # point = torch.where(img < 0.5, torch.zeros_like(img), point)
        else:
            r1 = F.conv2d(img, self.k1, stride=1, padding=1)
            r2 = F.conv2d(img, self.k2, stride=1, padding=1)
            r3 = F.conv2d(img, self.k3, stride=1, padding=1)
            r4 = F.conv2d(img, self.k4, stride=1, padding=1)

            # 避免除零错误：当conv_result=0时，结果设为0；否则取倒数
            r1 = torch.where(r1 != 0, 1.0 / r1, torch.zeros_like(r1))
            r2 = torch.where(r2 != 0, 1.0 / r2, torch.zeros_like(r2))
            r3 = torch.where(r3 != 0, 1.0 / r3, torch.zeros_like(r3))
            r4 = torch.where(r4 != 0, 1.0 / r4, torch.zeros_like(r4))

            point = r1 + r2 + r3 + r4

        return point

    def euler_line(self, img):
        row = self.avgpool_row_pad(img)
        row = torch.where(row != 0, 1.0 / row, torch.zeros_like(row))
        # row = 1 / row
        row = self.avgpool_row(row)
        # row = torch.where(img < 0.5, torch.zeros_like(img), row)

        col = self.avgpool_col_pad(img)
        col = torch.where(col != 0, 1.0 / col, torch.zeros_like(col))
        # col = 1 / col
        col = self.avgpool_col(col)
        # col = torch.where(img < 0.5, torch.zeros_like(img), col)

        return (row + col)

    def euler_face(self, img):
        # face = torch.where(img >= 0.5, 1.0 / img, torch.zeros_like(img))
        face = torch.where(img != 0, 1.0 / img, torch.zeros_like(img))
        # face = torch.ones_like(img, device=img.device)

        return face

    def euler_char(self, img, conn_type):
        mask = torch.where(img >= 0.5, img, torch.zeros_like(img))

        euler = img * (self.euler_point(img, conn_type) - self.euler_line(img) + self.euler_face(img))  # * mask
        return euler

    def euler_loss(self, img, gt):
        loss = 0

        img0 = img.clone()
        gt0 = gt.clone()
        batch_size = img0.shape[0]

        return self.huber(self.euler_char(img0[:, 1, ...].unsqueeze(1), 8),
                          self.euler_char(gt0[:, 1, ...].unsqueeze(1), 8)) + self.huber(
            self.euler_char(img0[:, 0, ...].unsqueeze(1), 4),
            self.euler_char(gt0[:, 0, ...].unsqueeze(1), 4))

        # return self.huber(self.euler_char(img0, 8), self.euler_char(gt0, 8))

    def euler_area(self, img, gt, t):
        img0 = img.clone()
        gt0 = gt.clone()
        bs, c, h, w = img.shape

        euler_img_fore = self.euler_char(img0)
        euler_gt_fore = self.euler_char(gt0)

        euler_img_back = self.euler_char(1 - img0)
        euler_gt_back = self.euler_char(1 - gt0)

        euler_img = t * euler_img_fore + (1 - t) * euler_img_back
        euler_gt = t * euler_gt_fore + (1 - t) * euler_gt_back

        # x_down = self.avgpool_area(torch.abs(euler_img - euler_gt))
        # weight = F.interpolate(x_down, size=(h, w), mode='nearest')

        weight = torch.abs(euler_img - euler_gt)

        # return weight / torch.amax(weight, dim=(2, 3), keepdim=True)
        return weight

    def dual_grad(self, img, gt, t=0.5):
        bs, c, h, w = img.shape

        img_fore = img[:, 1, ...].unsqueeze(1)
        gt_fore = gt[:, 1, ...].unsqueeze(1)

        point_grad = self.euler_point(img_fore) - img_fore * (self.euler_point(img_fore) ** 2)
        line_grad = self.euler_line(img_fore) - img_fore * (self.euler_line(img_fore) ** 2)
        face_grad = self.euler_face(img_fore) - img_fore * (self.euler_face(img_fore) ** 2)
        # print(img_fore.shape)

        euler_grad_fore = point_grad - line_grad + face_grad

        img_back = img[:, 0, ...].unsqueeze(1)
        gt_back = gt[:, 0, ...].unsqueeze(1)

        point_grad = self.euler_point(img_back) - img_back * (self.euler_point(img_back) ** 2)
        line_grad = self.euler_line(img_back) - img_back * (self.euler_line(img_back) ** 2)
        face_grad = self.euler_face(img_back) - img_back * (self.euler_face(img_back) ** 2)

        euler_grad_back = point_grad - line_grad + face_grad
        # print(euler_grad_back.shape)

        euler_grad = torch.concat([t * euler_grad_back, (1 - t) * euler_grad_fore], dim=1)

        # x_down = self.avgpool_area(self.euler_char(img) - self.euler_char(gt))
        # weight = F.interpolate(x_down, size=(h, w), mode='nearest')

        # weight = torch.concat([t * (self.euler_char(img_back) - self.euler_char(gt_back)),
        #                        (1 - t) * (self.euler_char(img_fore) - self.euler_char(gt_fore))], dim=1)
        # weight = torch.softmax(weight, dim=1)
        # weight = (weight>=0.5).float()
        weight = self.euler_char(img) - self.euler_char(gt)
        weight[weight > 0] = 1
        weight[weight < 0] = -1
        # print(torch.max(weight))

        grad = euler_grad
        # grad = img - gt
        # print(torch.min(grad))
        grad = grad * weight

        return grad

    def dual_loss(self, img, gt, pred_grad):
        img0 = img.clone()
        gt0 = gt.clone()
        pred_grad0 = pred_grad
        # pred_grad0 = F.log_softmax(pred_grad0, dim=1)
        # pred_grad0 = torch.softmax(pred_grad0, dim=1)

        return self.ce(pred_grad0, torch.softmax(self.dual_grad(img0, gt0, t=0.95), dim=1))

        # return self.mse(pred_grad0, img0-gt0)


if __name__ == '__main__':
    device = 'cpu'
    u = torch.rand((1, 1, 3, 3))
    u[0, 0, 0, 0] = 0.6
    u[0, 0, 0, 1] = 0.0
    u[0, 0, 0, 2] = 0
    u[0, 0, 1, 0] = 0
    u[0, 0, 1, 1] = 0
    u[0, 0, 1, 2] = 0.1
    u[0, 0, 2, 0] = 0.1
    u[0, 0, 2, 1] = 0.4
    u[0, 0, 2, 2] = 0.2
    u_b = (u >= 0.5).float()
    print(u)

    # sp = SimplepointLayer(1, device, 1)
    el = EulerLayer(1)
    r = el.euler_char(u, 8)
    # r = sp(u, 5)
    print(r)
