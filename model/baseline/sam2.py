# model with sam2_img_encoder and mask_decoder just for img seg , and add CCS blocks
import logging
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.distributed
from torchvision.transforms import Normalize, Resize, ToTensor
from model.sam2.modeling.sam.mask_decoder import MaskDecoder
from model.sam2.modeling.sam.prompt_encoder import PromptEncoder
from model.sam2.modeling.sam.transformer import TwoWayTransformer
from model.sam2.utils.transforms import SAM2Transforms
from train_utils import skeletonize


class SAM2B(torch.nn.Module):
    def __init__(
            self,
            image_encoder,
            image_size=1024,
            backbone_stride=16,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
    ):
        super().__init__()
        self._features = None
        self._orig_hw = None
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.multimask_output_in_sam = multimask_output_in_sam
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # Part 0: the preprocess transformer
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((image_size, image_size)),
                Normalize(self.mean, self.std),
            )
        )
        self._transforms = SAM2Transforms(
            resolution=self.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 1, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(1),
        # )
        self.maxpool = nn.MaxPool2d(15, 1, 7)
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.hidden_dim = image_encoder.neck.d_model

        # Part 2:SAM-style mask decoder for the final mask output
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.hidden_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=False,
            pred_obj_scores_mlp=False,
            use_multimask_token_for_obj_ptr=False,
        )

        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_image(self, input_image: torch.Tensor):
        backbone_out = self.image_encoder(input_image)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def forward(
            self, image
    ):
        """
        batched_input(list(dict)):A list over input information

        """
        self._orig_hw = image.shape[-2:]
        input_image = self.transforms(image).contiguous()

        # ---------image encoder------------------
        backbone_out = self.forward_image(input_image)

        _, vision_feats, _, _ = self._prepare_backbone_features(backbone_out)
        feats = [feat.permute(1, 2, 0).view(input_image.shape[0], -1, *feat_size)
                 for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                 ][::-1]  # list[(B,32,256,256),(B,64,128,128),(B,256,64,64)]
        high_res_features = feats[:-1]
        image_embed = feats[-1]
        # ------------prompt encoder---------------------

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        sparse_embeddings = sparse_embeddings.repeat(image_embed.shape[0], 1, 1)
        dense_embeddings = dense_embeddings.repeat(image_embed.shape[0], 1, 1, 1)

        # -----------mask decoder-----------------------
        low_res_masks, _, _, _ = self.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output_in_sam,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution

        if self.multimask_output_in_sam:
            o = self._transforms.postprocess_masks(
                low_res_masks[:,0,...].unsqueeze(1), self._orig_hw[-1]
            )
            v = self._transforms.postprocess_masks(
                low_res_masks[:,1,...].unsqueeze(1), self._orig_hw[-1]
            )
            sp = skeletonize.Skel_type(5, v.device, 1, 1, 's')
            o = o + (self.alpha ** 2) * v * self.maxpool(sp(torch.sigmoid(v)))
            return o, v
        else:
            o = self._transforms.postprocess_masks(
                low_res_masks, self._orig_hw[-1]
            )
            return o
