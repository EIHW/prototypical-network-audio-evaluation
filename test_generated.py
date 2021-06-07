import argparse
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.nn as nn
import torch

from utils import AverageMeter, compute_accuracy, euclidean_dist, mkdir
from torch.utils.data import DataLoader
from samplers.episodic_batch_sampler import EpisodicBatchSampler
from dataloaders.spec_loader import Spec
from dataloaders.gen_spec_loader import GenSpec
from models.final_convnet import ConvNet
from models.identity import Identity


# dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
# torch.zeros(2, 2, dtype=dtype)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

model_names.append('default_convnet')

parser = argparse.ArgumentParser(description='Pytorch Prototypical Networks Testing')
parser.add_argument('--train_dir', type=str, help='path to training data (default: none)')
parser.add_argument('--test_dir', type=str, metavar='train_dir', help='path to validation data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--evaluation_name', type=str, help='Evaluation name')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--cpu', default=False, action='store_true', help='CPU mode')
parser.add_argument('--checkpoint', type=str, help='model checkpoint path')
parser.add_argument('--results_name', type=str, help='name of the results csv')
parser.add_argument('--n_episodes', default=None, type=int, help='Number of episodes to average')
parser.add_argument('--n_way', default=None, type=int, help='Number of classes per episode')
parser.add_argument('--n_support', default=None, type=int, help='Number of support samples per class')
parser.add_argument('--n_query', default=None, type=int, help='Number of query samples')
parser.add_argument('--test_gen', default=None, type=str, help='name of cv file for test data generated')
parser.add_argument('--support_ori', default=None, type=str, help='name of csv file for support set original data')


# CHANGE HERE FOR STORED EMBEDDINGS 
PATH_EMBEDDINGS = 'embeddings/'   


def main():
    args = parser.parse_args()
    global results_path
    results_path = os.path.join('evaluations', args.evaluation_name)
    mkdir(results_path)
    options = vars(args)
    save_options_dir = os.path.join(results_path, 'options.txt')

    with open(save_options_dir, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(options.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # Create model
    print("=> creating model '{}'".format(args.arch))

    if args.arch == 'default_convnet':
        model = ConvNet()
    else:
        model = models.__dict__[args.arch]()

        if args.out_dim is not None:
            lin = nn.Linear(model.fc.in_features, args.out_dim)
            model.fc = lin
        else:
            model.fc = Identity()
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' )".format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

    if not args.cpu:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    model.eval()
    cudnn.benchmark = True
    # Testing data
    test_dataset = GenSpec(args.test_gen)
    val_dataset = Spec(args.support_ori)
    episodes_n_per_len = len(test_dataset)
    episodes_n_per_len = args.n_episodes
    print(f'episodes: {episodes_n_per_len}')
    test_sampler = EpisodicBatchSampler(test_dataset.labels, episodes_n_per_len, args.n_way, args.n_support + args.n_query)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                             num_workers=args.workers, pin_memory=True)

    val_sampler = EpisodicBatchSampler(val_dataset.labels, args.n_episodes, args.n_way, args.n_support + args.n_query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=args.workers, pin_memory=True)

    test(test_loader, val_loader, model, args,episodes_n_per_len)

def test(test_loader, val_loader, model, args,episodes_n_per_len):
    print('Testing...')
    losses = AverageMeter()
    accuracy = AverageMeter()
    predictions = np.array([])
    labels_list = np.array([])
    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        emb_pla,emb_col,emb_joi,emb_tri = [],[],[],[]

        for n_episode, batch in enumerate(zip(test_loader, val_loader)):
            print(f"episode_n {n_episode+1}")
            data, _ = [_ for _ in batch[0]]
            val_data, _ = [_ for _ in batch[1]]

            p = args.n_support * args.n_way
            data_support = val_data[:p]
            data_query = data[p:]
            # Compute class prototypes (n_way, output_dim)

            embedding_vectors = model(data_support).reshape(args.n_support, args.n_way, -1)
            class_prototypes = embedding_vectors.mean(dim=0)
            # embedding_vectors_array = (embedding_vectors.data).cpu().numpy()
            
            class_embed = {0 : emb_pla,1 : emb_col,2 : emb_joi,3 : emb_tri}
            for i in range(0,args.n_way):
                out = embedding_vectors[:, i, :]
                class_embed[i].append(out)

            labels = torch.arange(args.n_way).repeat(args.n_query)
            labels = labels.type(torch.LongTensor)
            # Compute loss and metrics
            logits = euclidean_dist(model(data_query), class_prototypes)
            loss = F.cross_entropy(logits, labels)
            acc = compute_accuracy(logits, labels)
            acc, f1 = compute_accuracy(logits, labels)
            # Record loss and accuracy
            losses.update(loss.item(), data_query.size(0))
            accuracy.update(f1, data_query.size(0))

            pred = torch.argmax(logits, dim=1)

            pred = (pred.data).cpu().numpy()
            true = (labels.data).cpu().numpy()

            predictions = np.append(predictions, pred)
            labels_list = np.append(labels_list, true)
        # SAVE PREDICTIONS
        data = {'true': labels_list,
                'predictions': predictions}
        pred_paths = os.path.join('predictions', args.evaluation_name)
        if not os.path.isdir(pred_paths):
            os.mkdir(pred_paths)
        df = pd.DataFrame(data, columns=['true', 'predictions'])
        df.to_csv(f'{pred_paths}/{args.test_gen[:-4]}_test_aug_eps{episodes_n_per_len}_{accuracy.avg:.3f}.csv', index=False)

        print('Test Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Test Accuracy {accuracy.val:.3f}, f1 avg: {accuracy.avg:.3f}\t'.format(loss=losses, accuracy=accuracy))
    
    # SAVE EMBEDDINGS
    emb_paths = os.path.join(PATH_EMBEDDINGS, args.evaluation_name)
    if not os.path.isdir(emb_paths):
        os.mkdir(emb_paths)

    with open(f'{emb_paths}/{args.test_gen[:-4]}_testaug_pla_emb_eps{episodes_n_per_len}_{accuracy.avg:.3f}.txt', 'wb') as f:
        for a in emb_pla:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.test_gen[:-4]}_testaug_col_emb_eps{episodes_n_per_len}_{accuracy.avg:.3f}.txt', 'wb') as f:
        for a in emb_col:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.test_gen[:-4]}_testaug_tri_emb_eps{episodes_n_per_len}_{accuracy.avg:.3f}.txt', 'wb') as f:
        for a in emb_tri:
            np.savetxt(f, a)
    with open(f'{emb_paths}/{args.test_gen[:-4]}_testaug_joi_emb_eps{episodes_n_per_len}_{accuracy.avg:.3f}.txt', 'wb') as f:
        for a in emb_joi:
            np.savetxt(f, a)
    return losses.avg, accuracy.avg

if __name__ == '__main__':
    main()
