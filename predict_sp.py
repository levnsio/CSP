import os
import sys

sys.path.append("..")
import time
import datetime
import torch
import numpy as np
import random
from PIL import Image
from build_model import build_model
from torchvision import transforms
import yaml
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from train_utils.skeletonize import Skel_type

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
    alpha = args.alpha

    # t = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(f"./save_weights/{d}/{t}/{alpha}", exist_ok=True)
    os.makedirs(f"./results_sp/{d}/{t}", exist_ok=True)
    sp = Skel_type(10, device, 1, 1, 's')

    n = 'img'
    data_path = os.path.join(args.data_path, f'0_{d}', f'{z}', f'{n}')

    if t == 'sam2':
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             ])
    else:
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])

    for weights in os.listdir(f"./save_weights/{d}/{t}/{alpha}/weights"):
        weights_path = f"./save_weights/{d}/{t}/{alpha}/weights/" + weights

        # load weights
        model = build_model(model_name=f"{t}", device=device, in_channel=in_channel, classes=num_classes, checkpoint=weights_path)

        # model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False)['model'])
        # model.load_。state_dict(torch.load(weights_path, map_location='cpu')['model'])

        for i in os.listdir(data_path):
            # print(i)
            img_index = i.split(".")[0]

            img_path = os.path.join(data_path, i)

            model.to(device)
            original_img = Image.open(img_path).convert('RGB')
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)
                t_start = time_synchronized()
                output, v = model(img.to(device))
                t_end = time_synchronized()
                print("inference time: {}".format(t_end - t_start))
                if num_classes == 1:
                    output = torch.sigmoid(output)
                    prediction = (output > 0.5).float().squeeze()
                    pred_sp = (sp(torch.sigmoid(v)) > 0.5).float().squeeze()
                    pred_v = (torch.sigmoid(v) > 0.5).float().squeeze()
                else:
                    prediction = output.argmax(1).squeeze(0)
                    pred_sp = sp.argmax(1).squeeze(0)

                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction[prediction == 1] = 255
                mask = Image.fromarray(prediction)

                pred_sp = pred_sp.to("cpu").numpy().astype(np.uint8)
                pred_sp[pred_sp == 1] = 255
                mask_sp = Image.fromarray(pred_sp)

                pred_v = pred_v.to("cpu").numpy().astype(np.uint8)
                pred_v[pred_v == 1] = 255
                mask_v = Image.fromarray(pred_v)

                os.makedirs(f"./results_sp/{d}/{t}/{weights.replace('.pth', '')}", exist_ok=True)
                os.makedirs(f"./results_sp/{d}/{t}/{weights.replace('.pth', '_sp')}", exist_ok=True)
                os.makedirs(f"./results_sp/{d}/{t}/{weights.replace('.pth', '_v')}", exist_ok=True)

                mask.save(os.path.join(f"./results_sp/{d}/{t}/" + weights.replace('.pth', ''), img_index + '.png'))
                mask_sp.save(os.path.join(f"./results_sp/{d}/{t}/" + weights.replace('.pth', '_sp'), img_index + '.png'))
                mask_v.save(os.path.join(f"./results_sp/{d}/{t}/" + weights.replace('.pth', '_v'), img_index + '.png'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch segment training")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument("--model", default="deeplab", type=str, help="model")
    parser.add_argument("--data-path", default="./data/", help="data root")
    parser.add_argument("--device", default="cuda:1", help="predicting device")
    parser.add_argument("--z", default="test", help="predicting data")
    parser.add_argument("--alpha", default=0.0, type=float, help="topo weights")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    set_seed(314)
    args = parse_args()
    main(args)
