import torch
import torch.nn as nn


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


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient

    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100, epsilon=1e-6):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index, epsilon=epsilon)


def cal_sp(x: torch.Tensor, sp: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    d = 0
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = sp[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(x_i, ignore_index)

            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]

        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = inter

        d += (inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_sp(x: torch.Tensor, mor: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    clprec = 0.
    for channel in range(x.shape[1]):
        clprec += cal_sp(x[:, channel, ...], mor[:, channel, ...], ignore_index, epsilon)

    return clprec / x.shape[1]


def cal_sp_dice(pred, sp_pred, gt, sp_gt, smooth=1e-6, ignore_index=-100):

    spsens = multiclass_sp(pred, sp_gt, ignore_index=ignore_index, epsilon=smooth)
    spprec = multiclass_sp(gt, sp_pred, ignore_index=ignore_index, epsilon=smooth)
    sp_dice = 1. - 2.0 * (spprec * spsens) / (spprec + spsens)

    return sp_dice
