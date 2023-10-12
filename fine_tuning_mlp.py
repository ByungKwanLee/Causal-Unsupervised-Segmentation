import argparse

import torch.nn.init
from tqdm import tqdm
from utils.utils import *
from modules.segment_module import transform, untransform, compute_modularity_based_codebook
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from loader.dataloader import dataloader
from torch.cuda.amp import autocast, GradScaler
from loader.netloader import network_loader, segment_mlp_loader, cluster_mlp_loader

cudnn.benchmark = True
scaler = GradScaler()

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_clean():
    dist.destroy_process_group()


@Wrapper.EpochPrint
def train(args, net, segment, cluster, train_loader, optimizer_segment, optimizer_cluster):
    global counter
    segment.train()

    total_acc = 0
    total_loss = 0
    total_loss_linear = 0
    total_loss_mod = 0


    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, batch in prog_bar:

        # optimizer
        with autocast():

            # image and label and self supervised feature
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            # intermediate features
            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat, segment.dropout)

            # computing modularity based codebook
            loss_mod = compute_modularity_based_codebook(cluster.cluster_probe, seg_feat_ema, grid=args.grid)

            # linear probe loss
            linear_logits = segment.linear(seg_feat_ema)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            flat_linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, args.n_classes)
            flat_label = label.reshape(-1)
            flat_label_mask = (flat_label >= 0) & (flat_label < args.n_classes)
            loss_linear = F.cross_entropy(flat_linear_logits[flat_label_mask], flat_label[flat_label_mask])

            # loss
            loss = loss_linear + loss_mod

        # optimizer
        optimizer_segment.zero_grad()
        optimizer_cluster.zero_grad()
        scaler.scale(loss).backward()
        if args.dataset=='cityscapes':
            scaler.unscale_(optimizer_segment)
            torch.nn.utils.clip_grad_norm_(segment.parameters(), 1)
        elif args.dataset=='cocostuff27':
            scaler.unscale_(optimizer_segment)
            torch.nn.utils.clip_grad_norm_(segment.parameters(), 0.1)
        scaler.step(optimizer_segment)
        scaler.step(optimizer_cluster)
        scaler.update()

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.reshape(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # loss check
        total_loss += loss.item()
        total_loss_linear += loss_linear.item()
        total_loss_mod += loss_mod.item()

        # real-time print
        desc = f'[Train] Loss: {total_loss / (idx + 1):.2f}={total_loss_linear / (idx + 1):.2f}{total_loss_mod / (idx + 1):.2f}'
        desc += f' ACC: {100. * total_acc / (idx + 1):.1f}%'
        prog_bar.set_description(desc, refresh=True)

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()


@Wrapper.TestPrint
def test(args, net, segment, cluster, nice, test_loader):
    global counter_test
    segment.eval()

    total_acc = 0
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        img = batch["img"].cuda()
        label = batch["label"].cuda()

        # intermediate feature
        with autocast():

            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat)

            # linear probe loss
            linear_logits = segment.linear(seg_feat_ema)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            flat_label = label.reshape(-1)
            flat_label_mask = (flat_label >= 0) & (flat_label < args.n_classes)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat_ema), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster
            cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), inference=True)

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.reshape(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # nice evaluation
        _, desc_nice = nice.eval(cluster_preds, label)

        # real-time print
        desc = f'[TEST] Acc (Linear): {100. * total_acc / (idx + 1):.1f}% | {desc_nice}'
        prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()

    # Interrupt for sync GPU Process
    if args.distributed: dist.barrier()


def main(rank, args, ngpus_per_node):

    # setup ddp process
    if args.distributed: ddp_setup(args, rank, ngpus_per_node)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # print argparse
    print_argparse(args, rank)

    # dataset loader
    train_loader, test_loader, sampler = dataloader(args)

    # network loader
    net = network_loader(args, rank)
    segment = segment_mlp_loader(args, rank)
    cluster = cluster_mlp_loader(args, rank)

    # distributed parsing
    if args.distributed: net = net.module; segment = segment.module; cluster = cluster.module

    # optimizer
    if args.dataset=='cityscapes':
        optimizer_segment = torch.optim.Adam(segment.parameters(), lr=1e-3 * ngpus_per_node)
        optimizer_cluster = torch.optim.Adam(cluster.parameters(), lr=1e-3 * ngpus_per_node)
    else:
        optimizer_segment = torch.optim.Adam(segment.parameters(), lr=1e-3 * ngpus_per_node, weight_decay=1e-4)
        optimizer_cluster = torch.optim.Adam(cluster.parameters(), lr=1e-3 * ngpus_per_node)
    
    # scheduler
    scheduler_segment = torch.optim.lr_scheduler.StepLR(optimizer_segment, step_size=2, gamma=0.5)
    scheduler_cluster = torch.optim.lr_scheduler.StepLR(optimizer_cluster, step_size=2, gamma=0.5)

    # evaluation
    nice = NiceTool(args.n_classes)

    ###################################################################################
    # First, run train_mediator.py
    path, is_exist = pickle_path_and_exist(args)

    # early save for time
    if is_exist:
        # load
        codebook = np.load(path)
        cb = torch.from_numpy(codebook).cuda()
        cluster.codebook.data = cb
        cluster.codebook.requires_grad = False

        # print successful loading modularity
        rprint(f'Modularity {path} loaded', rank)

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return
    ###################################################################################


    # train
    for epoch in range(args.epoch):

        # for shuffle
        if args.distributed: sampler.set_epoch(epoch)


        # train
        train(
            epoch,  # for decorator
            rank,  # for decorator
            args,
            net,
            segment,
            cluster,
            train_loader,
            optimizer_segment,
            optimizer_cluster)

        test(
            epoch, # for decorator
            rank, # for decorator
            args,
            net,
            segment,
            cluster,
            nice,
            test_loader)

        scheduler_segment.step()
        scheduler_cluster.step()

        if (rank == 0):
            x = segment.state_dict()
            baseline = args.ckpt.split('/')[-1].split('.')[0]

            # filepath hierarchy
            check_dir(f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}')

            # save path
            y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/segment_mlp.pth'
            torch.save(segment.state_dict(), y)

            y = f'CAUSE/{args.dataset}/{baseline}/{args.num_codebook}/cluster_mlp.pth'
            torch.save(cluster.state_dict(), y)
            print(f'-----------------TEST Epoch {epoch}: SAVING CHECKPOINT IN {y}-----------------')

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

    # Closing DDP
    if args.distributed: dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--NAME-TAG', default='CAUSE-MLP', type=str)
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets/', type=str)
    parser.add_argument('--dataset', default='cocostuff27', type=str)
    parser.add_argument('--ckpt', default='checkpoint/dino_vit_base_8.pth', type=str)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--distributed', default=True, type=str2bool)
    parser.add_argument('--load_segment', default=True, type=str2bool)
    parser.add_argument('--load_cluster', default=False, type=str2bool)
    parser.add_argument('--train_resolution', default=224, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)

    # DDP
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--port', default='12355', type=str)
    
    # codebook parameter
    parser.add_argument('--grid', default='yes', type=str2bool)
    parser.add_argument('--num_codebook', default=2048, type=int)

    # model parameter
    parser.add_argument('--reduced_dim', default=90, type=int)
    parser.add_argument('--projection_dim', default=2048, type=int)

    args = parser.parse_args()

    if 'dinov2' in args.ckpt:
        args.test_resolution=322
    if 'small' in args.ckpt:
        args.dim=384
    elif 'base' in args.ckpt:
        args.dim=768

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    if args.distributed:
        # cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # multiprocess spawn
        mp.spawn(main, args=(args, ngpus_per_node), nprocs=ngpus_per_node, join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        main(rank=gpu_list[0], args=args, ngpus_per_node=1)
