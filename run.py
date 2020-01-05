import torch
import os
import numpy as np
import random
import argparse
import hashnet

from loguru import logger
from data.data_loader import load_data


def run():
    # Load config
    args = load_config()
    logger.add('logs/{}_model_{}_code_{}_alpha_{}.log'.format(
            args.dataset,
            args.arch,
            args.code_length,
            args.alpha,
        ), 
        rotation='500 MB', 
        level='INFO',
    )
    logger.info(args)

    # Set seed
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    train_dataloader, query_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.batch_size,
        args.num_workers,
    )

    # Training
    checkpoint = hashnet.train(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        args.arch,
        args.code_length,
        args.device,
        args.lr,
        args.max_iter,
        args.alpha,
        args.topk,
        args.evaluate_interval,
    )
    logger.info('[code_length:{}][map:{:.4f}]'.format(args.code_length, checkpoint['map']))

    # Save checkpoint
    torch.save(
        checkpoint, 
        os.path.join('checkpoints', '{}_model_{}_code_{}_alpha_{}_map_{:.4f}.pt'.format(
            args.dataset, 
            args.arch, 
            args.code_length, 
            args.alpha, 
            checkpoint['map']),
        )
    )


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='HashNet_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--code-length', type=int,
                        help='Binary hash code length.')
    parser.add_argument('--arch', default='alexnet', type=str,
                        help='CNN model name.(default: alexnet)')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Batch size.(default: 256)')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--max-iter', default=300, type=int,
                        help='Number of iterations.(default: 300)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--alpha', default=1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')
    parser.add_argument('--evaluate-interval', default=10, type=int,
                        help='Evaluation interval.(default: 10)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    return args


if __name__ == '__main__':
    run()

