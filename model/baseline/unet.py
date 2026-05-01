from typing import List
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 classes: int = 2,
                 decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name='resnet101',
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=classes,
            # encoder_weights=None  # 不使用预训练权重
        )

        # self.e_head = SegmentationHead(
        #     in_channels=decoder_channels[-1],
        #     out_channels=classes,
        #     kernel_size=3
        # )


    def forward(self, img):
        """
        img: torch.Tensor(N,3,H,W)
        """
        features = self.model.encoder(img)  # 获取中间特征
        x = self.model.decoder(features)  # 解码
        o = self.model.segmentation_head(x)  # 原分割头输出

        return o


if __name__ == '__main__':
    # 创建模型
    model = UNet()
    # print(model)

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total number of trainable parameters {trainable_num}')
