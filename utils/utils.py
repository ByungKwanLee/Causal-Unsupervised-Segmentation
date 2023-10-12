from torchvision import transforms
import os
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool

invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])
Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

is_sym = lambda x: (x.transpose(1, 0) == x).all().item()

import datetime
class Wrapper(object):
    @staticmethod
    def InitializePrint(func):
        def wrapper(rank, *args, **kwards):
            rprint(f'-------------Initialize VQ-VAE-------------', rank)
            func(*args, **kwards)
        return wrapper
    @staticmethod
    def TestPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------TEST EPOCH: {epoch+1}-------------', rank)
            return func(*args, **kwards)
        return wrapper
    @staticmethod
    def EpochPrint(func):
        def wrapper(epoch, rank, *args, **kwards):
            rprint(f'-------------TRAIN EPOCH: {epoch+1}-------------', rank)
            func(*args, **kwards)
        return wrapper
    @staticmethod
    def KmeansPrint(func):
        def wrapper(rank, *args, **kwards):
            rprint(f'-------------K-Means Clustering-------------', rank)
            func(*args, **kwards)
        return wrapper
    @staticmethod
    def TimePrint(func):
        def wrapper(*args, **kwargs):
            start = datetime.datetime.now()
            out = func(*args, **kwargs)
            end = datetime.datetime.now()
            print(f'[{func.__name__}] Time: {(end - start).total_seconds():.2f}sec')
            return out
        return wrapper

def pickle_path_and_exist(args):
    from os.path import exists
    baseline = args.ckpt.split('/')[-1].split('.')[0]
    check_dir(f'CAUSE/{args.dataset}')
    check_dir(f'CAUSE/{args.dataset}/modularity')
    check_dir(f'CAUSE/{args.dataset}/modularity/{baseline}')
    check_dir(f'CAUSE/{args.dataset}/modularity/{baseline}/{args.num_codebook}')
    filepath = f'CAUSE/{args.dataset}/modularity/{baseline}/{args.num_codebook}/modular.npy'
    return filepath, exists(filepath)

def freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

def no_freeze(net):
    # net eval and freeze
    net.eval()
    for param in net.parameters():
        param.requires_grad = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_all(args, ind, img, label, cluster_preds, crf_preds, hungarian_preds, cmap, is_tr=False):
    baseline = args.ckpt.split('/')[-1].split('.')[0]
    y = f'{args.num_codebook}'
    check_dir(f'results')
    if is_tr:
        root = os.path.join('results', args.dataset, 'TR', baseline, y)
    else:
        root = os.path.join('results', args.dataset, 'MLP', baseline, y)

    check_dir(f'{root}/imgs')
    check_dir(f'{root}/labels')
    check_dir(f'{root}/kmeans')
    check_dir(f'{root}/crfs')
    check_dir(f'{root}/hungarians')
    # save image
    for id, i in [(id, x.item()) for id, x in enumerate(list(ind))]:
        torchvision.utils.save_image(invTrans(img)[id].cpu(), f'{root}/imgs/imgs_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[label[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/labels/labels_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[cluster_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/kmeans/kmeans_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[crf_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/crfs/crfs_{i}.png')
        torchvision.utils.save_image(torch.from_numpy(cmap[hungarian_preds[id].cpu()]).permute(2, 0, 1),
                                     f'{root}/hungarians/hungarians_{i}.png')

def rprint(msg, rank=0):
    if rank==0: print(msg)

def num_param(f):
    out = 0
    for param in f.head.parameters():
        out += param.numel()
    return out

def imshow(img):
    a = 255 * invTrans(img).permute(1, 2, 0).cpu().numpy()
    plt.imshow(a.astype(np.int64))
    plt.show()

def plot(x):
    plt.plot(x.cpu().numpy())
    plt.show()

def cmshow(img):
    # color map
    cmap = create_cityscapes_colormap()
    plt.imshow(cmap[img.cpu().numpy()])
    plt.show()

def getCMap(n_classes=27, cmapName='jet'):

    # Get jet color map from Matlab
    labelCount = n_classes
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    return cmap

def ckpt_to_name(ckpt):
    name = ckpt.split('/')[-1].split('_')[0]
    return name

def ckpt_to_arch(ckpt):
    arch = ckpt.split('/')[-1].split('.')[0]
    return arch

def print_argparse(args, rank=0):
    dict = vars(args)
    if rank == 0:
        print('------------------Configurations------------------')
        for key in dict.keys(): print("{}: {}".format(key, dict[key]))
        print('-------------------------------------------------')


"""

BELOW are STEGO Fucntion

"""

from collections import OrderedDict
class NiceTool(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.histogram = torch.zeros((self.n_classes, self.n_classes)).cuda()

    def scores(self, label_trues, label_preds):
        mask = (label_trues >= 0) & (label_trues < self.n_classes) & (label_preds >= 0) & (label_preds < self.n_classes)  # Exclude unlabelled data.
        hist = torch.bincount(self.n_classes * label_trues[mask] + label_preds[mask], \
                              minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes).t().cuda()
        return hist

    def eval(self, pred, label):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        self.histogram += self.scores(label, pred)

        self.assignments = linear_sum_assignment(self.histogram.cpu(), maximize=True)
        hist = self.histogram[np.argsort(self.assignments[1]), :]

        tp = torch.diag(hist)
        fp = torch.sum(hist, dim=0) - tp
        fn = torch.sum(hist, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(hist)

        # metric
        metric_dict = OrderedDict({"mIoU": iou[~torch.isnan(iou)].mean().item() * 100,
                       # "Precision per Class (%)": prc * 100,
                       "mAP": prc[~torch.isnan(prc)].mean().item() * 100,
                       "Acc": opc.item() * 100})


        self.metric_dict_by_class = OrderedDict({"mIoU": iou * 100,
                       # "Precision per Class (%)": prc * 100,
                       "mAP": prc * 100,
                       "Acc": (torch.diag(hist) / hist.sum(dim=1)) * 100})


        # generate desc
        sentence = ''
        for key, value in metric_dict.items():
            if type(value) == torch.Tensor: continue
            sentence += f'[{key}]: {value:.1f}, '
        return metric_dict, sentence

    def reset(self):
        self.histogram = torch.zeros((self.n_classes, self.n_classes)).cuda()

    def do_hungarian(self, clusters):
        return torch.tensor(self.assignments[1])[clusters.cpu()]


from scipy.optimize import linear_sum_assignment
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor, max_iter: int):
    MAX_ITER = max_iter
    POS_W = 3
    POS_XY_STD = 1
    Bi_W = 4
    Bi_XY_STD = 67
    Bi_RGB_STD = 3

    image = np.array(VF.to_pil_image(invTrans(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q

def _apply_crf(tup, max_iter):
    return dense_crf(tup[0], tup[1], max_iter=max_iter)

def do_crf(pool, img_tensor, prob_tensor, max_iter=10):
    from functools import partial
    outputs = pool.map(partial(_apply_crf, max_iter=max_iter), zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

def create_pascal_label_colormap():
    def bit_get(val, idx):
        return (val >> idx) & 1
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap / 255

def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors, dtype=int) / 255



class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)



from torchvision.transforms import InterpolationMode
# DATA Transformation
def get_cococity_transform(res, is_label):

    if is_label:
        return transforms.Compose([transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
                          transforms.CenterCrop(res),
                          ToTargetTensor()])
    else:
        return transforms.Compose([transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
                          transforms.CenterCrop(res),
                          transforms.ToTensor()])

# DATA Transformation
def get_pascal_transform(res, is_label):
    if is_label:
        return transforms.Compose([transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
                          ToTargetTensor()])
    else:
        return transforms.Compose([transforms.Resize((res, res), interpolation=InterpolationMode.NEAREST),
                          transforms.ToTensor()])