import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import torch.utils.data
import torch.multiprocessing as mp
import torch.optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch
import warnings
import time
import random
import argparse
import os
import numpy as np
from models.final_convnet import ConvNet
from models.identity import Identity
from utils import AverageMeter, save_checkpoint, compute_accuracy, weights_init_xavier, mkdir, euclidean_dist
from samplers.episodic_batch_sampler import EpisodicBatchSampler
from dataloaders.spec_loader import Spec
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

dtype = torch.float
torch.zeros(2, 2, dtype=dtype)

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append('default_convnet')

parser = argparse.ArgumentParser(description='PyTorch Prototypical Networks Training')
parser.add_argument('--train_dir', type=str, help='path to training data (default: none)')
parser.add_argument('--val_dir', type=str, metavar='train_dir', help='path to validation data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_name', type=str, help='model_filename')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-s', '--image_size', default=224, type=int, help='Image Size to load images')
parser.add_argument('--n_episodes_train', default=200, type=int, help='Number of episodes per epoch at train')
parser.add_argument('--n_way_train', default=10, type=int, help='Number of classes per episode at train')
parser.add_argument('--n_query_train', default=1, type=int, help='Number of query samples at train')
parser.add_argument('--n_support', default=5, type=int, help='Number of support samples')
parser.add_argument('--n_episodes_val', default=200, type=int, help='Number of episodes per epoch at validation')
parser.add_argument('--n_way_val', default=10, type=int, help='Number of classes per episode at validation')
parser.add_argument('--n_query_val', default=1, type=int, help='Number of query samples at validation')
parser.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use: "adam" or "sgd"')
parser.add_argument('--step_size', default=30, type=int, help='Scheduler step size')
parser.add_argument('--gamma', default=0.1, type=float, help='Scheduler gamma')
parser.add_argument('--alpha', default=0.0, type=float, help='Controls the contribution from past prototypes in next'
                                                             'episodes')
parser.add_argument('--out_dim', default=None, type=int, help='Output embedding dimension')
parser.add_argument('--train_csv', default=None, type=str, help='train_csv file')

best_acc1 = 0

def main():
    args = parser.parse_args()
    global results_dir
    results_dir = os.path.join('models_trained', args.model_name)
    mkdir(results_dir)

    options = vars(args)
    save_options_dir = os.path.join(results_dir, 'options.txt')


    with open(save_options_dir, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(options.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Create model
    if args.arch == 'default_convnet':
        model = ConvNet()
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        if args.out_dim is not None:
            lin = nn.Linear(model.fc.in_features, args.out_dim)
            weights_init_xavier(lin)
            model.fc = lin
        else:
            model.fc = Identity()

    print('Number of parameters: ', sum([p.numel() for p in model.parameters()]))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model)

    # Define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    else:
        raise ValueError('Optimizer should be "sgd" or "adam"')

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')

            # print(checkpoint())
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            # altered for loading from check-pint
            state_dict = checkpoint['state_dict']
            for k, v in state_dict.items():
                if 'module' not in k:
                    print(k)
                    k = 'module.' + k
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
#            optimizer.load_state_dict(torch.load(os.path.join(args.resume), map_location='cpu'))
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint  # dereference seems crucial  
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_dataset = Spec(args.train_csv)
    train_sampler = EpisodicBatchSampler(train_dataset.labels, args.n_episodes_train, args.n_way_train,
                                         args.n_support + args.n_query_train)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.workers,
                              pin_memory=True)
    print(f'{len(train_loader)} loader_shape')
    val_dataset = Spec('devel_ori_specs_meta')
    val_sampler = EpisodicBatchSampler(val_dataset.labels, args.n_episodes_val, args.n_way_val,
                                       args.n_support + args.n_query_val)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=args.workers,
                            pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, args,0)
        return
    # added hard epoch for checkpoint resume

    for epoch in range(0, args.epochs):
        lr_scheduler.step()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        loss_t, acc_t = train(train_loader, model, optimizer, epoch, args)

        # Evaluate on validation set
        loss_val, acc1 = validate(val_loader, model, args,epoch)

        dict_metrics = {'loss_training': loss_t, 'loss_validation': loss_val,
                        'acc_training': acc_t, 'acc_validation': acc1}

        for key in dict_metrics:
            with open(os.path.join(results_dir, key + '.txt'), "a+") as myfile:
                myfile.write(str(dict_metrics[key]) + '\n')

        # Remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print('Saving model...')
            if args.gpu is None:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, results_dir)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, results_dir)

def train(train_loader, model, optimizer, epoch, args):
    print("Training epoch %d" % epoch)
    episode_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # Switch to train mode
    model.train()
    end = time.time()
    optimizer.zero_grad()

    emb_pla = []
    emb_col = []
    emb_joi = []
    emb_tri = []
    # Iterate over episodes
    for n_episode, batch in enumerate(train_loader, 1):
        print(f"episode_n {n_episode}")
        data_time.update(time.time() - end)
        data, _ = [_ for _ in batch]
        p = args.n_support * args.n_way_train
        data_support, data_query = data[:p], data[p:]

        # Compute class prototypes (n_way, output_dim)
        if n_episode > 1 and args.alpha > 0.0:
            embedding_vectors = args.alpha * class_prototypes + (1 - args.alpha) * \
                model(data_support).reshape(args.n_support, args.n_way_train, -1)
            class_prototypes = embedding_vectors.mean(dim=0)
        else:
            embedding_vectors = model(data_support).reshape(args.n_support, args.n_way_train, -1)
            class_prototypes = embedding_vectors.mean(dim=0)

        embedding_vectors = embedding_vectors.cpu().detach().numpy()

        for i in range(0, args.n_way_train):
            # print(i)
            out = embedding_vectors[:, i, :]
            # .mean(dim=1)
            # print(f'out: {out.shape}')
            if i == 0:
                emb_pla.append(out)
            if i == 1:
                emb_col.append(out)
            if i == 2:
                emb_joi.append(out)
            if i == 3:
                emb_tri.append(out)

        # # Generate labels (n_way, n_query)
        labels = torch.arange(args.n_way_train).repeat(args.n_query_train)
        labels = labels.type(torch.LongTensor)
        #
        # # Compute loss and metrics
        logits = euclidean_dist(model(data_query), class_prototypes)
        loss = F.cross_entropy(logits, labels)
        acc, f1 = compute_accuracy(logits, labels)

        # # Record loss and accuracy
        losses.update(loss.item(), data_query.size(0))
        accuracy.update(f1, data_query.size(0))
        #
        # # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Free the graph
        if args.alpha > 0.0:
             class_prototypes = class_prototypes.detach()
             embedding_vectors = embedding_vectors.detach()
             torch.cuda.empty_cache()

        else:
             class_prototypes = None
             embedding_vectors = None
             torch.cuda.empty_cache()

        # # Measure elapsed time
        episode_time.update(time.time() - end)
        end = time.time()
        #
        if n_episode % args.print_freq == 0:
             print('Epoch: [{0}][{1}/{2}]\t'
                   'Episode Time {episode_time.val:.3f} ({episode_time.avg:.3f})\t'
                   'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                   .format(
                       epoch, n_episode, args.n_episodes_train, episode_time=episode_time,
                       data_time=data_time, loss=losses, accuracy=accuracy))

    emb_paths = os.path.join('embeddings/', args.model_name)
    if not os.path.isdir(emb_paths):
        os.mkdir(emb_paths)

    with open(f'{emb_paths}/{args.train_csv[:-4]}_train_ep{epoch}_ori_pla.txt', 'wb') as f:
        for a in emb_pla:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.train_csv[:-4]}train_ep{epoch}_ori_col_emb.txt', 'wb') as f:
        for a in emb_col:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.train_csv[:-4]}train_ep{epoch}_ori_tri.txt', 'wb') as f:
        for a in emb_tri:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.train_csv[:-4]}train_ep{epoch}_ori_joi.txt', 'wb') as f:
        for a in emb_joi:
            np.savetxt(f, a)

    return losses.avg, accuracy.avg,





def validate(val_loader, model, args,epoch):
    print('Validating...')
    losses = AverageMeter()
    accuracy = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        emb_pla, emb_col, emb_joi, emb_tri = [], [], [], []

        for n_episode, batch in enumerate(val_loader, 1):
            data, _ = [_ for _ in batch]
            p = args.n_support * args.n_way_val
            data_support, data_query = data[:p], data[p:]


            # Compute class prototypes (n_way, output_dim)
            class_prototypes = model(data_support).reshape(args.n_support, args.n_way_val, -1).mean(dim=0)

            # Generate labels (n_way, n_query)
            labels = torch.arange(args.n_way_val).repeat(args.n_query_val)
            labels = labels.type(torch.LongTensor)

            # Compute loss and metrics
            logits = euclidean_dist(model(data_query), class_prototypes)
            loss = F.cross_entropy(logits, labels)
            acc, f1 = compute_accuracy(logits, labels)

            # Record loss and accuracy
            losses.update(loss.item(), data_query.size(0))
            accuracy.update(f1, data_query.size(0))

        print('Validation Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Validation Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(loss=losses, accuracy=accuracy))

    return losses.avg, accuracy.avg


if __name__ == '__main__':
    main()
