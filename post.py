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
from torchvision import transforms
from thop import profile, clever_format


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


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    t = args.model
    d = config['dataset']
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    z = args.z

    in_channel = config['in_channel']
    num_classes = config['num_classes']  # 算背景
    mean = tuple(config['mean'])
    std = tuple(config['std'])
    base_size = config['base_size']
    alpha = args.alpha

    # t = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    # os.makedirs(f"./save_weights/{d}/{t}/{alpha}", exist_ok=True)
    # os.makedirs(f"./results_{z}/{d}/{t}", exist_ok=True)
    s_type = 'b'
    n = 'img'
    data_path = os.path.join(args.data_path, f'0_{d}', f'{z}', f'{n}')
    # gt_path = os.path.join(args.data_path, f'0_{d}', f'{z}', 'mask')

    os.makedirs(f"./results_skel_/{d}/{t}/" + s_type, exist_ok=True)

    # data_transform = T.Compose([
    #     T.pad_to_square(0),
    #     T.RandomResize(base_size),
    #     T.ToTensor(),
    #     T.Normalize(mean=mean, std=std),
    # ])

    data_transform = T.Compose([T.ToTensor()])

    for weights in os.listdir(f"./save_weights/{d}/{t}/{alpha}/weights"):
        weights_path = f"./save_weights/{d}/{t}/{alpha}/weights/" + weights
        model = build_model(model_name=f"{t}", device=device, in_channel=in_channel, classes=num_classes, checkpoint=weights_path)
        model.to(device)

        for i in os.listdir(data_path):
            # print(i)
            img_index = i.split(".")[0]

            img_path = os.path.join(data_path, i)
            # mask_path = os.path.join(gt_path, i.split('.')[0] + '_gt.png')

            original_img = Image.open(img_path).convert('RGB')
            # gt = Image.open(mask_path).convert('L')
            gt = original_img

            img, gt = data_transform(original_img, gt)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            gt = torch.unsqueeze(gt, dim=0)
            gt = torch.unsqueeze(gt, dim=0)/255
            # print(img.shape)
            # print(gt.shape)
            sp = Skel_type(5, device, 1, 1, s_type)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)
                t_start = time_synchronized()
                output,_ = model(img.to(device))
                # p = std.p(gt)
                # s = sp(gt)
                # u = torch.softmax(output, dim=1)
                s = sp(torch.sigmoid(output))
                output2 = s
                # output2 = torch.sigmoid(output) + nn.functional.max_pool2d(s * (gt - torch.sigmoid(output)), 13, 1, 6) * gt
                # output2 = torch.sigmoid(output + 8 * s)
                # output2 = torch.sigmoid(output + 10 * nn.functional.max_pool2d(s,13,1,6) * gt)

                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))
                if num_classes == 1:
                    output = torch.sigmoid(output)
                    prediction = (output > 0.5).float().squeeze()
                    # output2 = torch.sigmoid(output2)
                    prediction2 = (output2 > 0.5).float().squeeze()

                else:
                    prediction = output.argmax(1).squeeze(0)
                    prediction2 = output2.argmax(1).squeeze(0)
                    # prediction2 = torch.sigmoid(output2[0,1,...]) * 255

                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255
                mask = Image.fromarray(prediction)

                prediction2 = prediction2.to("cpu").numpy().astype(np.uint8)
                prediction2[prediction2 == 1] = 255
                mask2 = Image.fromarray(prediction2)

                if not os.path.exists(f"./results_skel_/{d}/{t}/" + s_type):
                    os.mkdir(f"./results_skel_/{d}/{t}/" + s_type)
                mask.save(
                    os.path.join(f"./results_skel_/{d}/{t}/" + s_type, img_index + '.png'))
                mask2.save(
                    os.path.join(f"./results_skel_/{d}/{t}/" + s_type, img_index + '_post.png'))

                # if not os.path.exists(f"./results_skel_/{d}/{t}/"):
                #     os.mkdir(f"./results_skel_/{d}/{t}/")
                # mask.save(
                #     os.path.join(f"./results_skel_/{d}/{t}/", img_index + '.png'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch segment training")
    parser.add_argument('--config', default="config/drive.yaml", type=str, required=True,
                        help='Path to the config file')
    parser.add_argument("--model", default="seg", type=str, help="model")
    parser.add_argument("--data-path", default="./data/", help="data root")
    parser.add_argument("--device", default="cuda:0", help="predicting device")
    parser.add_argument("--z", default="test", help="predicting data")
    parser.add_argument("--alpha", default=0.0, type=float, help="topo weights")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    set_seed(314)
    args = parse_args()
    main(args)
