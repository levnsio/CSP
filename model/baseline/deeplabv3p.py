import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


class DeepLabV3P(nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet101",
                 in_channels: int = 3,
                 classes: int = 2,
                 decoder_channels: int = 256):
        super(DeepLabV3P, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=decoder_channels,
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
    model = DeepLabV3P()
    # print(model)

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total number of trainable parameters {trainable_num}')
