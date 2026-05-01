# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import logging
import os


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None, in_channel=3, out_channel=3):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        in_channel=in_channel,
        out_channel=out_channel,
    )


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        in_channel=3,
        out_channel=3,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            in_chans=in_channel,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=out_channel,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict, strict=False)
    return sam


def load_ckpt(model: torch.nn.Module, ckpt: str):
    """
    Load the checkpoint file of a pre-trained model.

    Parameters:
    model (torch.nn.Module): The model instance to load the weights into.
    ckpt (str): The path to the pre-trained model checkpoint file.

    Returns:
    torch.nn.Module: The model instance with pre-trained weights loaded.
    """
    # Open the checkpoint file and load its contents
    assert os.path.exists(ckpt), f"Checkpoint file {ckpt} not found."
    with open(ckpt, "rb") as f:
        state_dict = torch.load(f, map_location='cpu', weights_only=False)
    state_dict = state_dict['model']
    # get state_dict for this model
    model_dict = model.state_dict()
    fliter_state_dict = {}

    # Iterate over the weights in the checkpoint
    logging.info(f"{model.__class__.__name__} starts loading checkpoint")
    for k, x in model_dict.items():
        if k in state_dict:
            # Check if the shape of the weight matches the shape of the current model's weight
            if x.shape == state_dict[k].shape:
                fliter_state_dict[k] = state_dict[k]
            else:
                logging.warning(f"Skip loading parameter: {k}, "
                                f"required shape: {x.shape}, "
                                f"loaded shape: {state_dict[k].shape}")

    # Update the current model's weight dictionary with the filtered weights
    logging.info(f"Loaded {len(fliter_state_dict)} / {len(model_dict)} parameters from pretrained weight file: {ckpt}")
    model_dict.update(fliter_state_dict)
    model.load_state_dict(model_dict)

    return model
