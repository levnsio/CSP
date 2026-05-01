import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import torchvision.transforms as T
import torch.nn.functional as F
from model.segment_anything.modeling import image_encoder, prompt_encoder, mask_decoder
# from ..module.e_std import ESTDLayer, SoftEdge
from torchvision.transforms.functional import resize, to_pil_image


# Constants
# IMAGE_WIDTH = 1024


class SAM(nn.Module):
    def __init__(
            self, device,
            image_encoder: image_encoder,
            mask_decoder: mask_decoder,
            prompt_encoder: prompt_encoder,
            pixel_mean: List[float] = [0.485, 0.456, 0.406],
            pixel_std: List[float] = [0.229, 0.224, 0.225],
    ):
        super(SAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # self.softmax = nn.Softmax2d()
        # self.softmax = nn.Sigmoid()
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.device = device

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # freeze image encoder
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return newh, neww

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        newh, neww = self.get_preprocess_shape(h, w, self.image_encoder.img_size)

        image_scale = []
        for i in range(x.shape[0]):
            # 获取单张图片scaling后的结果
            img_sample = np.array(resize(to_pil_image(x[i]), [newh, neww]))
            image_scale.append(img_sample)
        # image_scale = np.array(resize(to_pil_image(image), target_size))

        input_image_torch = torch.as_tensor(np.array(image_scale), device=self.device)
        # 由BxhxWxC# 转变成Bxcxhxw的tensor数据
        input_image_torch = input_image_torch.permute(0, 3, 1, 2).contiguous()

        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        padh = self.image_encoder.img_size - newh
        padw = self.image_encoder.img_size - neww
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        return input_image_torch

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.size befor padding after scaling
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, img):
        img_pad = self.preprocess(img)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )  # 提示编码

        # 图像编码
        image_embedding = self.image_encoder(img_pad)  # (N, 256, 64, 64)
        # 解码获得低分辨率掩码
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (N, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (0, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (0, 256, 64, 64)
            multimask_output=True,
        )
        # 上采样
        o = self.postprocess_masks(low_res_masks,
                                   img_pad.shape[-2:], img.shape[-2:])  # (N, 3, 1024, 1024)
        del image_embedding, low_res_masks  # 删除无用变量

        # if self.std == 0:
        #     o = outputs
        #     # o = outputs[:, 0, :, :].unsqueeze(1)
        #     e = self.edge(self.softmax(o))
        # else:
        #     if self.e == 0:
        #         o = self.estd(outputs, self.e)
        #         # o = self.estd(outputs[:, 0, :, :].unsqueeze(1), self.e)
        #         e = self.edge(self.softmax(o))
        #     else:
        #         split_idx = outputs.size(1) // 2  # 确保 c 是偶数
        #         o = outputs[:, :split_idx, :, :]  # (N, 1, 1024, 1024)
        #         e = outputs[:, split_idx:, :, :]  # (N, 1, 1024, 1024)
        #
        #         # o = outputs[:, 0, :, :].unsqueeze(1)  # (N, 1, 1024, 1024)
        #         # e = outputs[:, 1, :, :].unsqueeze(1)  # (N, 1, 1024, 1024)
        #
        #         e = self.edge(self.softmax(e))
        #         o = self.estd(o, e)
        return o


if __name__ == '__main__':
    x = torch.randn(2, 6, 3, 3)
    split_idx = x.size(1) // 2  # 6//2=3

    # 分割张量
    o = x[:, :split_idx, :, :]  # 前3个通道
    w = x[:, split_idx:, :, :]  # 后3个通道

    print("Original shape:", x.shape)
    print("o shape:", o.shape)  # [2, 3, 3, 3]
    print("w shape:", w.shape)  # [2, 3, 3, 3]
