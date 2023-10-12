import os
import torch
import argparse
from PIL import Image
from os.path import join
from utils.utils import *
from torch.utils.data import DataLoader
from loader.dataloader import ContrastiveSegDataset
from torchvision.transforms.functional import five_crop, ten_crop
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

class RandomCropComputer(Dataset):

    @staticmethod
    def _get_size(img, crop_ratio):
        if len(img.shape) == 3:
            return [int(img.shape[1] * crop_ratio), int(img.shape[2] * crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * crop_ratio), int(img.shape[1] * crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def __init__(self, args, dataset_name, img_set, crop_type, crop_ratio):
        self.pytorch_data_dir = args.data_dir
        self.crop_ratio = crop_ratio

        if crop_type == 'five':
            crop_func = lambda x: five_crop(x, self._get_size(x, crop_ratio))
        elif crop_type == 'double':
            crop_ratio = 0
            crop_func = lambda x: ten_crop(x, self._get_size(x, 0.5))\
                                + ten_crop(x, self._get_size(x, 0.8))
        elif crop_type == 'super':
            crop_ratio = 0
            crop_func = lambda x: ten_crop(x, self._get_size(x, 0.3))\
                                + ten_crop(x, self._get_size(x, 0.4))\
                                + ten_crop(x, self._get_size(x, 0.5))\
                                + ten_crop(x, self._get_size(x, 0.6))\
                                + ten_crop(x, self._get_size(x, 0.7))

        if args.dataset=='coco171':
            self.save_dir = join(
                args.data_dir, 'cocostuff', "cropped", "coco171_{}_crop_{}".format(crop_type, crop_ratio))
        elif args.dataset=='coco81':
            self.save_dir = join(
                args.data_dir, 'cocostuff', "cropped", "coco81_{}_crop_{}".format(crop_type, crop_ratio))
        else:
            self.save_dir = join(
                args.data_dir, dataset_name, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.args = args

        self.img_dir = join(self.save_dir, "img", img_set)
        self.label_dir = join(self.save_dir, "label", img_set)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # train dataset
        self.dataset = ContrastiveSegDataset(
            pytorch_data_dir=args.data_dir,
            dataset_name=args.dataset,
            crop_type=None,
            image_set=img_set,
            transform=T.ToTensor(),
            target_transform=ToTargetTensor(),
            extra_transform=crop_func
        )
        
    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


def my_app():

    # fetch args
    parser = argparse.ArgumentParser()

    # fixed parameter
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)

    # dataset and baseline
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='cocostuff27', type=str)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--distributed', default='false', type=str2bool)
    parser.add_argument('--crop_type', default='five', type=str)
    parser.add_argument('--crop_ratio', default=0.5, type=float)

    args = parser.parse_args()
    
    # setting gpu id of this process
    torch.cuda.set_device(args.gpu)

    counter = 0
    dataset = RandomCropComputer(args, args.dataset, "train", args.crop_type, args.crop_ratio)
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers, collate_fn=lambda l: l)
    for batch in tqdm(loader):
        imgs = batch[0]['img']
        labels = batch[0]['label']
        for img, label in zip(imgs, labels):
            img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)
            Image.fromarray(img_arr).save(join(dataset.img_dir, "{}.jpg".format(counter)), 'JPEG')
            Image.fromarray(label_arr).save(join(dataset.label_dir, "{}.png".format(counter)), 'PNG')
            counter+=1

if __name__ == "__main__":
    my_app()