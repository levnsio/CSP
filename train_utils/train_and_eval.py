import torch
from torch import nn
import train_utils.distributed_utils as utils
from .Loss import criterion, criterion_ssp
import numpy as np
import math


def evaluate(model, data_loader, device, num_classes, ignore_index=-100):
    model.eval()
    if num_classes == 1:
        confmat = utils.ConfusionMatrix(num_classes + 1)
    else:
        confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=ignore_index)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'eval:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 200, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if num_classes == 1:
                confmat.update(target.flatten(), (torch.sigmoid(output) > 0.5).int().flatten())
            else:
                confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        acc_global, acc, rec, dic, iou = confmat.compute()

    return confmat, dice.value.item(), acc_global.item(), iou.mean().item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, alpha, beta, half_size, ignore_index, skel_type, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(enabled=scaler is not None, device_type="cuda"):

            output = model(image)
            loss = criterion(output, target, alpha, beta, num_classes, half_size, ignore_index, skel_type=skel_type)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate_ssp(model, data_loader, device, num_classes, ignore_index=-100):
    model.eval()
    if num_classes == 1:
        confmat = utils.ConfusionMatrix(num_classes + 1)
    else:
        confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=ignore_index)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'eval:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 200, header):
            image, target = image.to(device), target.to(device)
            output, v = model(image)
            if num_classes == 1:
                confmat.update(target.flatten(), (torch.sigmoid(output) > 0.5).int().flatten())
            else:
                confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        acc_global, acc, rec, dic, iou = confmat.compute()

    return confmat, dice.value.item(), acc_global.item(), iou.mean().item()


def train_one_epoch_ssp(model, optimizer, data_loader, device, epoch, num_classes,
                        lr_scheduler, alpha, beta, half_size, ignore_index, skel_type, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast(enabled=scaler is not None, device_type="cuda"):

            output, v = model(image)
            loss = criterion_ssp(output, v, target, alpha, beta, num_classes, half_size, ignore_index,
                                 skel_type=skel_type)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=5,
                        warmup_factor=1e-3,
                        lr_decay=True):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """根据 step 数计算学习率倍率因子"""
        warmup_steps = warmup_epochs * num_step

        # Warmup 阶段（线性增长）
        if warmup and x <= warmup_steps:
            alpha = float(x) / max(1, warmup_steps)
            return warmup_factor * (1 - alpha) + alpha

        # 余弦退火阶段
        elif lr_decay:
            total_decay_steps = (epochs - warmup_epochs) * num_step
            progress = (x - warmup_steps) / max(1, total_decay_steps)
            progress = min(progress, 1.0)  # 限制在 [0,1] 范围内
            eta_min = 1e-8
            return eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

        # 不衰减
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
