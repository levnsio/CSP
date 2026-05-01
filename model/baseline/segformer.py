from typing import List
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


class SegFormer(nn.Module):
    def __init__(self,
                 encoder_name: str = "mit_b5",
                 in_channels: int = 3,
                 classes: int = 2,
                 decoder_channels: int = 256):
        super(SegFormer, self).__init__()
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            decoder_segmentation_channels=decoder_channels,
            # encoder_weights=None # 不使用预训练权重
        )

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
    model = SegFormer()
    # print(model)
    # 统计参数量
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total number of trainable parameters {trainable_num}')

    # 打印每一层的名称和参数
    # print("\nLayer details:")
    # for name, layer in model.named_modules():
    #     print(f"Layer: {name}")
    #     print(layer)
