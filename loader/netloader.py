from utils.utils import *
from modules.segment import Segment_MLP
from modules.segment import Segment_TR
from modules.segment_module import Cluster
from torch.nn.parallel import DistributedDataParallel

def network_loader(args, rank=0):
    # load network
    net = load_model(args.ckpt, rank).cuda()
    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[rank])
    freeze(net)
    return net

def cluster_mlp_loader(args, rank):
    cluster = Cluster(args).cuda()

    if args.load_cluster:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/cluster_mlp.pth'
        cluster.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Cluster] {y} loaded', rank)

    if args.distributed:
        cluster = DistributedDataParallel(cluster, device_ids=[rank])
    return cluster


def cluster_tr_loader(args, rank):
    cluster = Cluster(args).cuda()

    if args.load_cluster:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/cluster_tr.pth'
        cluster.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Cluster] {y} loaded', rank)

    if args.distributed:
        cluster = DistributedDataParallel(cluster, device_ids=[rank])
    return cluster

def segment_mlp_loader(args, rank=0):
    segment = Segment_MLP(args).cuda()

    if args.load_segment:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/segment_mlp.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Segment] {y} loaded', rank)

    if args.distributed:
        segment = DistributedDataParallel(segment, device_ids=[rank])

    return segment

def segment_tr_loader(args, rank=0):
    segment = Segment_TR(args).cuda()

    if args.load_segment:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/segment_tr.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Segment] {y} loaded', rank)

    if args.distributed:
        segment = DistributedDataParallel(segment, device_ids=[rank])

    return segment


def checkpoint_module(checkpoint, net):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    msg = net.load_state_dict(new_state_dict, strict=False)
    return msg

def load_model(ckpt, rank=0):
    # name and arch
    name = ckpt_to_name(ckpt)
    arch = ckpt_to_arch(ckpt)

    if name == "dino" or name == "mae":
        import models.dinomaevit as model
    elif name == "dinov2":
        import models.dinov2vit as model
    elif name == "ibot":
        import models.ibotvit as model
    elif name == "msn":
        import models.msnvit as model
    else:
        raise ValueError

    net = getattr(model, arch)()
    checkpoint = torch.load(ckpt, map_location=torch.device(f'cuda:{rank}'))
    if name == "mae":
        msg = net.load_state_dict(checkpoint["model"], strict=False)
    elif name == "dino":
        msg = net.load_state_dict(checkpoint, strict=False)
    elif name == "dinov2":
        msg = net.load_state_dict(checkpoint, strict=False)
    elif name == "ibot":
        msg = net.load_state_dict(checkpoint['state_dict'], strict=False)
    elif name == "msn":
        msg = checkpoint_module(checkpoint['target_encoder'], net)

    # check incompatible layer or variables
    rprint(msg, rank)

    return net
