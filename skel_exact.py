import os
import sys

sys.path.append("..")
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image
from build_model import build_model
from data_set import transforms as T
import yaml
# from train_utils import grad_f, div_f, Vis_Field, shape_field, DT, STDLayer, SimplepointLayer, MorphLayer
from train_utils import Skel_type
# from thop import profile, clever_format


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    skel_mode = 'm'
    d = 'ubw'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z = 'test'
    num_classes = 1
    iter_ = 10
    os.makedirs(f"./results_skel/{d}/{skel_mode}", exist_ok=True)

    data_path = "./data/"

    gt_path = os.path.join(data_path, f'0_{d}', f'{z}', 'mask')
    t = 0
    n = 0

    for i in os.listdir(gt_path):
        # print(i)
        img_index = i.split(".")[0]

        mask_path = os.path.join(gt_path, i)

        gt = Image.open(mask_path).convert('L')

        gt = np.array(gt)/255.
        gt = torch.tensor(gt)
        # expand batch dimension
        gt = torch.unsqueeze(gt, dim=0).to(device)
        gt = torch.unsqueeze(gt, dim=0).float()
        # gt = torch.concat([1 - gt, gt], dim=1)

        sp = Skel_type(iter_, device, 1, 1, skel_mode)
        # with torch.no_grad():
        img_height, img_width = gt.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        sp(init_img)
        t_start = time_synchronized()
        s = sp(gt)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        if num_classes == 1:
            prediction = (s > 0.5).float().squeeze()
            # output2 = torch.sigmoid(output2)

        else:
            prediction = s.argmax(1).squeeze(0)

        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        prediction[prediction == 1] = 255
        mask = Image.fromarray(prediction)

        mask.save(os.path.join(f"./results_skel/{d}/{skel_mode}", img_index + '.png'))
        t = t + t_end - t_start
        n = n + 1
    print("inference time: {}".format(t/n*1000))


if __name__ == '__main__':
    set_seed(314)
    main()
