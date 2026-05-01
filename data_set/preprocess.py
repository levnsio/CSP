from data_set import transforms as T
from torchvision import transforms


class Preset:
    def __init__(self, train, base_size=512, crop_size=320, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if train:
            trans = [T.pad_to_square(0)]  # 随机选取一个size，并按最小边等比例缩放
            max_size = int(base_size * 1.2)
            min_size = int(base_size * 0.8)
            trans.append(T.RandomResize(min_size, max_size)) # 随机选取一个size，并按最小边等比例缩放
            if hflip_prob > 0:  # 水平翻转
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            if vflip_prob > 0:  # 竖直翻转
                trans.append(T.RandomVerticalFlip(vflip_prob))
            trans.extend([
                T.RandomCrop(crop_size),  # 随机裁剪
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.transforms = T.Compose(trans)  # 图像处理的操作打包赋给transforms

        else:
            self.transforms = T.Compose([
                T.pad_to_square(0),
                T.RandomResize(base_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class samPreset:
    def __init__(self, train, base_size=512, crop_size=320, hflip_prob=0.5, vflip_prob=0.5):
        if train:
            trans = [T.pad_to_square(0)]  # 随机选取一个size，并按最小边等比例缩放
            max_size = int(base_size * 1.2)
            min_size = int(base_size * 0.8)
            trans.append(T.RandomResize(min_size, max_size)) # 随机选取一个size，并按最小边等比例缩放
            if hflip_prob > 0:  # 水平翻转
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            if vflip_prob > 0:  # 竖直翻转
                trans.append(T.RandomVerticalFlip(vflip_prob))
            trans.extend([
                T.ToTensor()
            ])
            self.transforms = T.Compose(trans)  # 图像处理的操作打包赋给transforms

        else:
            self.transforms = T.Compose([
                T.pad_to_square(0),
                T.RandomResize(base_size),
                T.ToTensor()
            ])

    def __call__(self, img, target):
        return self.transforms(img, target)


