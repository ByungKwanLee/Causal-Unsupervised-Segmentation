import random
from os.path import join

import torch.multiprocessing
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets import VOCSegmentation


from utils.utils import *

from torch.utils.data.distributed import DistributedSampler


def dataloader(args, no_ddp_train_shuffle=True):

    if args.dataset == "cocostuff27":
        args.n_classes = 27
        get_transform = get_cococity_transform
    elif args.dataset == "cityscapes":
        args.n_classes = 27
        get_transform = get_cococity_transform
    elif args.dataset == "pascalvoc":
        args.n_classes = 21
        get_transform = get_pascal_transform
    elif args.dataset == "coco81":
        args.n_classes = 81
        get_transform = get_cococity_transform
    elif args.dataset == "coco171":
        args.n_classes = 171
        get_transform = get_cococity_transform

    # train dataset
    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name=args.dataset,
        crop_type="five",
        image_set="train",
        transform=get_transform(args.train_resolution, False),
        target_transform=get_transform(args.train_resolution, True),
    )

    if args.distributed: train_sampler = DistributedSampler(train_dataset, shuffle=True)

    # train loader
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=False if args.distributed else no_ddp_train_shuffle, num_workers=args.num_workers,
                              pin_memory=True, sampler=train_sampler if args.distributed else None)

    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=args.data_dir,
        dataset_name=args.dataset,
        crop_type=None,
        image_set="val",
        transform=get_transform(args.test_resolution, False),
        target_transform=get_transform(args.test_resolution, True),
    )

    if args.distributed: test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # test dataloader
    test_loader = DataLoader(test_dataset, args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, sampler=test_sampler if args.distributed else None)

    sampler = train_sampler if args.distributed else None

    return train_loader, test_loader, sampler



class Coco81(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super().__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.transform = transform
        self.label_transform = target_transform

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, "Coco164kFull_Stuff_Coarse.txt"), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 
                               6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 
                               12: 12, 13: 13, 14: 14, 15: 15, 
                               16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 
                               21: 21, 22: 22, 23: 23, 24: 24, 26: 25, 
                               27: 26, 30: 27, 31: 28, 32: 29, 33: 30, 
                               34: 31, 35: 32, 36: 33, 37: 34, 38: 35, 
                               39: 36, 40: 37, 41: 38, 42: 39, 43: 40, 
                               45: 41, 46: 42, 47: 43, 48: 44, 49: 45, 
                               50: 46, 51: 47, 52: 48, 53: 49, 54: 50, 
                               55: 51, 56: 52, 57: 53, 58: 54, 59: 55, 
                               60: 56, 61: 57, 62: 58, 63: 59, 64: 60, 
                               66: 61, 69: 62, 71: 63, 72: 64, 73: 65, 
                               74: 66, 75: 67, 76: 68, 77: 69, 78: 70, 
                               79: 71, 80: 72, 81: 73, 83: 74, 84: 75, 
                               85: 76, 86: 77, 87: 78, 88: 79, 89: 80, 
                               91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 
                               97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 
                               103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 
                               109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 
                               115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 
                               121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 
                               127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 
                               133: 0, 134: 0, 135: 0, 136: 0, 137: 0, 138: 0, 
                               139: 0, 140: 0, 141: 0, 142: 0, 143: 0, 144: 0, 
                               145: 0, 146: 0, 147: 0, 148: 0, 149: 0, 150: 0, 
                               151: 0, 152: 0, 153: 0, 154: 0, 155: 0, 156: 0, 
                               157: 0, 158: 0, 159: 0, 160: 0, 161: 0, 162: 0, 
                               163: 0, 164: 0, 165: 0, 166: 0, 167: 0, 168: 0, 
                               169: 0, 170: 0, 171: 0, 172: 0, 173: 0, 174: 0, 
                               175: 0, 176: 0, 177: 0, 178: 0, 179: 0, 180: 0, 181: 0, 255: -1}


    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1
        return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class Coco171(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super().__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.transform = transform
        self.label_transform = target_transform

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, "Coco164kFull_Stuff_Coarse.txt"), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {
                                0: 0,
                                1: 1,
                                2: 2,
                                3: 3,
                                4: 4,
                                5: 5,
                                6: 6,
                                7: 7,
                                8: 8,
                                9: 9,
                                10: 10,
                                12: 11,
                                13: 12,
                                14: 13,
                                15: 14,
                                16: 15,
                                17: 16,
                                18: 17,
                                19: 18,
                                20: 19,
                                21: 20,
                                22: 21,
                                23: 22,
                                24: 23,
                                26: 24,
                                27: 25,
                                30: 26,
                                31: 27,
                                32: 28,
                                33: 29,
                                34: 30,
                                35: 31,
                                36: 32,
                                37: 33,
                                38: 34,
                                39: 35,
                                40: 36,
                                41: 37,
                                42: 38,
                                43: 39,
                                45: 40,
                                46: 41,
                                47: 42,
                                48: 43,
                                49: 44,
                                50: 45,
                                51: 46,
                                52: 47,
                                53: 48,
                                54: 49,
                                55: 50,
                                56: 51,
                                57: 52,
                                58: 53,
                                59: 54,
                                60: 55,
                                61: 56,
                                62: 57,
                                63: 58,
                                64: 59,
                                66: 60,
                                69: 61,
                                71: 62,
                                72: 63,
                                73: 64,
                                74: 65,
                                75: 66,
                                76: 67,
                                77: 68,
                                78: 69,
                                79: 70,
                                80: 71,
                                81: 72,
                                83: 73,
                                84: 74,
                                85: 75,
                                86: 76,
                                87: 77,
                                88: 78,
                                89: 79,
                                91: 80,
                                92: 81,
                                93: 82,
                                94: 83,
                                95: 84,
                                96: 85,
                                97: 86,
                                98: 87,
                                99: 88,
                                100: 89,
                                101: 90,
                                102: 91,
                                103: 92,
                                104: 93,
                                105: 94,
                                106: 95,
                                107: 96,
                                108: 97,
                                109: 98,
                                110: 99,
                                111: 100,
                                112: 101,
                                113: 102,
                                114: 103,
                                115: 104,
                                116: 105,
                                117: 106,
                                118: 107,
                                119: 108,
                                120: 109,
                                121: 110,
                                122: 111,
                                123: 112,
                                124: 113,
                                125: 114,
                                126: 115,
                                127: 116,
                                128: 117,
                                129: 118,
                                130: 119,
                                131: 120,
                                132: 121,
                                133: 122,
                                134: 123,
                                135: 124,
                                136: 125,
                                137: 126,
                                138: 127,
                                139: 128,
                                140: 129,
                                141: 130,
                                142: 131,
                                143: 132,
                                144: 133,
                                145: 134,
                                146: 135,
                                147: 136,
                                148: 137,
                                149: 138,
                                150: 139,
                                151: 140,
                                152: 141,
                                153: 142,
                                154: 143,
                                155: 144,
                                156: 145,
                                157: 146,
                                158: 147,
                                159: 148,
                                160: 149,
                                161: 150,
                                162: 151,
                                163: 152,
                                164: 153,
                                165: 154,
                                166: 155,
                                167: 156,
                                168: 157,
                                169: 158,
                                170: 159,
                                171: 160,
                                172: 161,
                                173: 162,
                                174: 163,
                                175: 164,
                                176: 165,
                                177: 166,
                                178: 167,
                                179: 168,
                                180: 169,
                                181: 170,
                                255: -1
                            }


    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1
        return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None):
        super(Coco, self).__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return image, target.squeeze(0), mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform):
        super(CroppedDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = image_set
        if dataset_name=='coco171':
            self.root = join(root, "cocostuff", "cropped", "coco171_{}_crop_{}".format(crop_type, crop_ratio))
        elif dataset_name=='coco81':
            self.root = join(root, "cocostuff", "cropped", "coco81_{}_crop_{}".format(crop_type, crop_ratio))
        else:
            self.root = join(root, dataset_name, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = join(self.root, "img", self.split)
        self.label_dir = join(self.root, "label", self.split)
        self.num_images = len(os.listdir(self.img_dir))
        assert self.num_images == len(os.listdir(self.label_dir))

    def __getitem__(self, index):
        image = Image.open(join(self.img_dir, "{}.jpg".format(index))).convert('RGB')
        target = Image.open(join(self.label_dir, "{}.png".format(index)))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.target_transform(target)

        target = target - 1
        mask = target == -1
        return image, target.squeeze(0), mask

    def __len__(self):
        return self.num_images

class PascalVOC(VOCSegmentation):
    def __init__(self, root, year, image_set, download, transforms, target_transforms):
        super().__init__(root, year=year, image_set=image_set, download=download, transforms=transforms)
        self.target_transforms=target_transforms
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.masks[idx])

        if self.transforms is not None:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.transforms(image)
            self._set_seed(seed); label = self.target_transforms(label)
            label[label > 20] = -1
        return image, label.squeeze(0)
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def PascalVOCGenerator(root, image_set, transform, target_transform):
        return PascalVOC(join(root, "pascalvoc"), 
                    year='2012', 
                    image_set=image_set, 
                    download=False, 
                    transforms=transform,
                    target_transforms=target_transform)

class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 extra_transform=None,
                 ):
        super(ContrastiveSegDataset).__init__()
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.extra_transform = extra_transform
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


        # cityscapes, cocostuff27 
        if  dataset_name == "cityscapes" and crop_type is None:
            dataset_class = CityscapesSeg
            extra_args = dict()
        elif dataset_name == "cocostuff27" and crop_type is None:
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if image_set == "val":
                extra_args["subset"] = 7

        # cityscapes, cocostuff27 [Crop]
        elif dataset_name == "cityscapes" and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=0.5)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff", crop_type="five", crop_ratio=0.5)


        # coco-81, coco-171, pascalvoc
        elif dataset_name == "coco81" and crop_type is None:
            dataset_class = Coco81
            extra_args = dict()
        elif dataset_name == "coco171" and crop_type is None:
            dataset_class = Coco171
            extra_args = dict()
        elif dataset_name == "pascalvoc" and crop_type is None:
            dataset_class = PascalVOC.PascalVOCGenerator
            extra_args = dict()

        # coco-81, coco-171, pascalvoc [Crop]
        elif dataset_name == "coco81" and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="coco81", crop_type='double', crop_ratio=0)
        elif dataset_name == "coco171" and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="coco171", crop_type='double', crop_ratio=0)
        elif dataset_name == "pascalvoc" and crop_type is not None:
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="pascalvoc", crop_type='super', crop_ratio=0)
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        self._set_seed(seed)
        if self.extra_transform is not None:
            extra_trans = self.extra_transform
            self.normalize = lambda x: x
        else:
            extra_trans = lambda x: x

        ret = {
            "ind": ind,
            "img": self.normalize(extra_trans(pack[0])),
            "label": extra_trans(pack[1]),
        }

        return ret

