import torch
import argparse
import os
import numpy as np
import random
from loguru import logger

import harr

from data.data_loader import load_data
from model_loader import load_model
import warnings
warnings.filterwarnings("ignore")
def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
multi_labels_dataset = [
    'nus-wide-tc-10',
    'nus-wide-tc-21',
    'flickr25k',
]

num_features = {
    'alexnet': 4096,
    'vgg16': 4096,
}


def run():
    # Load configuration
    seed_torch()
    seed= 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    args = load_config()
    logger.add(os.path.join('logs', '{time}.log'), rotation="500 MB", level="INFO")
    logger.info(args)

    # Load dataset
    query_dataloader, train_dataloder, retrieval_dataloader, train_dataloader_wag = load_data(args.dataset,
                                                                        args.root,
                                                                        args.num_query,
                                                                        args.num_train,
                                                                        args.batch_size,
                                                                        args.num_workers,
                                                                        )

    multi_labels = args.dataset in multi_labels_dataset
    if args.train:
        harr.train(
            train_dataloder,
            query_dataloader,
            retrieval_dataloader,
            train_dataloader_wag,
            multi_labels,
            args.code_length,
            num_features[args.arch],
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.topk,
            args.threshold,
            args.eta_1,
            args.eta_2,
            args.temperature,
            args.evaluate_interval,
        )
    else:
        raise ValueError('Error configuration, please check your config, using "train", "resume" or "evaluate".')


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='HARR_PyTorch')
    parser.add_argument('-d', '--dataset', default = 'cifar-10',
                        help='Dataset name.')
    parser.add_argument('-r', '--root', default = '../../Dataset',
                        help='Path of dataset')
    parser.add_argument('-c', '--code-length', default=128, type=int,
                        help='Binary hash code length.(default: 64)')
    parser.add_argument('-T', '--max-iter', default=600, type=int,
                        help='Number of iterations.(default: 150)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        help='Learning rate.(default: 1e-3)')
    parser.add_argument('-q', '--num-query', default=10000, type=int,
                        help='Number of query data points.(default: 10000)')
    parser.add_argument('-t', '--num-train', default=70, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('-w', '--num-workers', default=4, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('-b', '--batch-size', default=48, type=int,
                        help='Batch size.(default: 24)')
    parser.add_argument('-a', '--arch', default='vgg16', type=str,
                        help='CNN architecture.(default: vgg16)')
    parser.add_argument('-k', '--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: -1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print log.')
    parser.add_argument('--train', action='store_true',
                        help='Training mode.')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluation mode.')
    parser.add_argument('-g', '--gpu', default=2, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('-e', '--evaluate-interval', default=50, type=int,
                        help='Interval of evaluation.(default: 500)')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='Hyper-parameter.(default:0.1)')
    parser.add_argument('--eta_1', default=0.9, type=float,
                        help='Hyper-parameter.(default:0.3)')
    parser.add_argument('--eta_2', default=0.5, type=float,
                        help='Hyper-parameter.(default:0.3)')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='Hyper-parameter in SimCLR .(default:0.5)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(args.gpu)

    return args


if __name__ == '__main__':
    run()
