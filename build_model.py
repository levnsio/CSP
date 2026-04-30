import os
import torch
import numpy as np
from model import SegFormer, DeepLabV3P, UNetPP, SAM
import logging
from model.segment_anything import sam_model_registry

from model.baseline.sam2 import SAM2B
from model.sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from model.sam2.modeling.backbones.hieradet import Hiera
from model.sam2.modeling.position_encoding import PositionEmbeddingSine


def build_model(model_name: str, device,
                checkpoint: str = None,
                in_channel: int = 3, classes: int = 2,
                origin_model: str = 'seg', **kwargs):
    if model_name == "seg":
        model = SegFormer(encoder_name=kwargs.get("encoder_name", "mit_b5"),
                          in_channels=kwargs.get("in_channels", in_channel),
                          classes=kwargs.get("classes", classes),
                          decoder_channels=kwargs.get("decoder_channels", 256))
    elif model_name == "deeplab":
        model = DeepLabV3P(encoder_name=kwargs.get("encoder_name", "resnet101"),
                           in_channels=kwargs.get("in_channels", in_channel),
                           classes=kwargs.get("classes", classes),
                           decoder_channels=kwargs.get("decoder_channels", 256))
    elif model_name == "unetpp":
        model = UNetPP(in_channels=kwargs.get("in_channels", in_channel),
                       classes=kwargs.get("classes", classes),
                       decoder_channels=kwargs.get("decoder_channels", (256, 128, 64, 32, 16)))
    elif model_name == "sam":

        weight_path = None
        # sam_model = sam_model_registry['vit_b'](checkpoint=weight_path)
        sam_model = sam_model_registry['vit_b'](checkpoint=weight_path, in_channel=in_channel, out_channel=classes)

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = True

        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = True

        model = SAM(image_encoder=sam_model.image_encoder,
                    mask_decoder=sam_model.mask_decoder,
                    prompt_encoder=sam_model.prompt_encoder,
                    device=device)

    elif model_name == "sam2":

        model = SAM2B(
            image_encoder=ImageEncoder(
                scalp=1,
                trunk=Hiera(embed_dim=112, num_heads=2),
                neck=FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                             d_model=256,
                             backbone_channel_list=[896, 448, 224, 112],
                             fpn_top_down_levels=[2, 3],
                             fpn_interp_model='nearest')),
            multimask_output_in_sam=True
        )

    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    if checkpoint is not None and os.path.exists(checkpoint):
        model = load_ckpt(model, checkpoint, model_name)

    return model


def load_ckpt(model: torch.nn.Module, ckpt: str, model_name: str):
    """
    Load the checkpoint file of a pre-trained model.

    Parameters:
    model (torch.nn.Module): The model instance to load the weights into.
    ckpt (str): The path to the pre-trained model checkpoint file.

    Returns:
    torch.nn.Module: The model instance with pre-trained weights loaded.
    """
    print(" load ckpt ")
    # Open the checkpoint file and load its contents
    assert os.path.exists(ckpt), f"Checkpoint file {ckpt} not found."
    with open(ckpt, "rb") as f:
        state_dict = torch.load(f, map_location='cpu', weights_only=False)
        # print(ckpt)
        # state_dict = torch.load(f, map_location='cpu')
    if model_name == 'sam':
        state_dict = state_dict
    else:
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

# if __name__ == "__main__":
#     # 设置logging的输出等级为INFO
#     logging.basicConfig(level=logging.INFO)
#     sam2model = build_sam2base(checkpoint="/home/xiej/data/models/sam2.1_hiera_base_plus.pt")
