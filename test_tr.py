import argparse

from tqdm import tqdm
from utils.utils import *
from modules.segment_module import transform, untransform
from loader.dataloader import dataloader
from torch.cuda.amp import autocast
from loader.netloader import network_loader, segment_tr_loader, cluster_tr_loader


def test(args, net, segment, cluster, nice, test_loader, cmap):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    with Pool(40) as pool:
        for _, batch in prog_bar:
            # image and label and self supervised feature
            ind = batch["ind"].cuda()
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            with autocast():
                # intermediate feature
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            seg_feat = transform(segment.head_ema(feat))
            seg_feat_flip = transform(segment.head_ema(feat_flip))
            seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster preds
            cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), crf=True)

            # crf
            crf_preds = do_crf(pool, img, cluster_preds).argmax(1).cuda()

            # nice evaluation
            _, desc_nice = nice.eval(crf_preds, label)

            # hungarian
            hungarian_preds = nice.do_hungarian(crf_preds)

            # save images
            save_all(args, ind, img, label, cluster_preds.argmax(dim=1), crf_preds, hungarian_preds, cmap, is_tr=True)

            # real-time print
            desc = f'{desc_nice}'
            prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()



def test_without_crf(args, net, segment, cluster, nice, test_loader):
    segment.eval()

    total_acc = 0
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        ind = batch["ind"].cuda()
        img = batch["img"].cuda()
        label = batch["label"].cuda()

        cmap = create_pascal_label_colormap()
        a = invTrans(img)[0].permute(1,2,0)
        b = cmap[label[0].cpu()]

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

            # nice evaluation
            _, desc_nice = nice.eval(cluster_preds, label)

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.reshape(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # real-time print
        desc = f'[TEST] Acc (Linear): {100. * total_acc / (idx + 1):.1f}% | {desc_nice}'
        prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()


def test_linear_without_crf(args, net, segment, nice, test_loader):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    with Pool(40) as pool:
        for _, batch in prog_bar:
            # image and label and self supervised feature
            ind = batch["ind"].cuda()
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            with autocast():
                # intermediate feature
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            seg_feat = transform(segment.head_ema(feat))
            seg_feat_flip = transform(segment.head_ema(feat_flip))
            seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat), label.shape[-2:], mode='bilinear', align_corners=False)

            # linear probe interp feat
            linear_logits = segment.linear(untransform(interp_seg_feat))

            # cluster preds
            cluster_preds = linear_logits.argmax(dim=1)

            # nice evaluation
            _, desc_nice = nice.eval(cluster_preds, label)

            # real-time print
            desc = f'{desc_nice}'
            prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()



def test_linear(args, net, segment, nice, test_loader):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    with Pool(40) as pool:
        for _, batch in prog_bar:
            # image and label and self supervised feature
            ind = batch["ind"].cuda()
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            with autocast():
                # intermediate feature
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            seg_feat = transform(segment.head_ema(feat))
            seg_feat_flip = transform(segment.head_ema(feat_flip))
            seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat), label.shape[-2:], mode='bilinear', align_corners=False)

            # linear probe interp feat
            linear_logits = segment.linear(untransform(interp_seg_feat))

            # cluster preds
            cluster_preds = torch.log_softmax(linear_logits, dim=1)

            # crf
            crf_preds = do_crf(pool, img, cluster_preds).argmax(1).cuda()

            # nice evaluation
            _, desc_nice = nice.eval(crf_preds, label)

            # real-time print
            desc = f'{desc_nice}'
            prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()


def main(rank, args):

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # print argparse
    print_argparse(args, rank=0)

    # dataset loader
    train_loader, test_loader, _ = dataloader(args, False)

    # network loader
    net = network_loader(args, rank)
    segment = segment_tr_loader(args, rank)
    cluster = cluster_tr_loader(args, rank)

    # evaluation
    nice = NiceTool(args.n_classes)

    # color map
    cmap = create_cityscapes_colormap() if args.dataset == 'cityscapes' else create_pascal_label_colormap()


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
        segment.head.codebook = cb
        segment.head_ema.codebook = cb

        # print successful loading modularity
        rprint(f'Modularity {path} loaded', rank)

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return
    ###################################################################################

    # param size
    print(f'# of Parameters: {num_param(segment)/10**6:.2f}(M)') 

    # post-processing with crf and hungarian matching
    test_without_crf(
        args,
        net,
        segment,
        cluster,
        nice,
        test_loader)

    # post-processing with crf and hungarian matching
    test(
        args,
        net,
        segment,
        cluster,
        nice,
        test_loader,
        cmap)
    
    # post-processing with crf and hungarian matching
    # test_linear_without_crf(
    #     args,
    #     net,
    #     segment,
    #     nice,
    #     test_loader)
    
    # test_linear(
    #     args,
    #     net,
    #     segment,
    #     nice,
    #     test_loader)


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    
    # model parameter
    parser.add_argument('--NAME-TAG', default='CAUSE-TR', type=str)
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='pascalvoc', type=str)
    parser.add_argument('--port', default='12355', type=str)
    parser.add_argument('--ckpt', default='checkpoint/dino_vit_small_8.pth', type=str)
    parser.add_argument('--distributed', default=False, type=str2bool)
    parser.add_argument('--load_segment', default=True, type=str2bool)
    parser.add_argument('--load_cluster', default=True, type=str2bool)
    parser.add_argument('--train_resolution', default=320, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)
    parser.add_argument('--gpu', default='4', type=str)
    parser.add_argument('--num_codebook', default=2048, type=int)

    # model parameter
    parser.add_argument('--reduced_dim', default=90, type=int)
    parser.add_argument('--projection_dim', default=2048, type=int)

    args = parser.parse_args()


    if 'dinov2' in args.ckpt:
        args.train_resolution=322
        args.test_resolution=322
    if 'small' in args.ckpt:
        args.dim=384
    elif 'base' in args.ckpt:
        args.dim=768
    args.num_queries=args.train_resolution**2 // int(args.ckpt.split('_')[-1].split('.')[0])**2
    

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    # first gpu index is activated once there are several gpu in args.gpu
    main(rank=gpu_list[0], args=args)