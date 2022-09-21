
import time
import os
import torch
import time
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
from torchvision import transforms
from pointnet2_ops import pointnet2_utils

from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import test_run_net as test_net
from utils import  parser,dist_utils, misc
from utils.logger import *
from utils.config import *

from tools import builder
from utils import misc, dist_utils

from utils.AverageMeter import AverageMeter

from datasets import data_transforms
from datasets import build_dataset_from_cfg

train_transforms = transforms.Compose(
    [
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)
import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./cfgs/pretrain.yaml", help='yaml config file')
    #./cfgs/finetune_scan_hardest.yaml
    #./cfgs/pretrain.yaml

    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', default=True, action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')

    # bn
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type=str, default='ScanObjectNN', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default="./experiments/pretrain/cfgs/ScanObjectNN/ckpt-last.pth",
                        help='test used ckpt path')  #
    # ./experiments/pretrain/cfgs/ScanObjectNN/ckpt-last.pth
    # ./pretrainedmodels/pretrain.pth
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--vote', action='store_true', default=False, help='vote acc')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=False,
                        help='finetune modelnet with pretrained weight')
    # if pretrain(finetune_model), False, otherwise True
    parser.add_argument('--scratch_model', action='store_true', default=False, help='training modelnet from scratch')
    parser.add_argument('--mode', choices=['easy', 'median', 'hard', None], default=None,
                        help='difficulty mode for shapenet')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both activate')
    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError('ckpts shouldnt be None while test mode')
    if args.finetune_model and args.ckpts is None:
        print('training from scratch')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,
                                        args.exp_name)
    # ./experiments/pretrain/cgs/ScanObjectNN/
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',
                                     args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args
def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger=logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)

    return acc
def model_builder(config):
    model = build_model_from_cfg(config)
    return model
class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict
def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed: #False
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle,
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)#

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:#fasle
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        #if args.ckpts is not None:
        if args.ckpts == None: #MAE=,Transformer!=
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            # print("taxonomy_ids",taxonomy_ids);
            # print("model_ids",model_ids);
            # print("data:",data)
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  # (B, N, 3)

            # import pdb; pdb.set_trace()
            points = train_transforms(points)
            print("hello, stephanie")
            ret = base_model(points);
            print("ret",ret.shape)

            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
                                                 logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
                                                'ckpt-best_vote', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
def get_config(args, logger=None):
    print("hello!")
    if args.resume:#False
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger)
        args.config = cfg_path

    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config

def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:#not false=true，进入
        if args.local_rank == 0:#=0，进入
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    print("config")
    config = get_config(args, logger = logger)

    # batch size
    if args.distributed:#False
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size 
    else:# this way
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):#False
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs 
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None: # this way
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed: #False
        assert args.local_rank == torch.distributed.get_rank() 

    if args.shot != -1:# Fasle
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold
        
    # run
    if args.test: #False
        test_net(args, config)
    else:
        # args.fineture_model==True; args.scratch_model==False
        if args.finetune_model or args.scratch_model:#True
            run_net(args, config, train_writer, val_writer)
        else: # if pretrain this way
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    # import os
    # os.environ["PATH"] = os.environ["PATH"]+":/usr/local/miniconda3/envs/mae/bin/ninja"
    main()
