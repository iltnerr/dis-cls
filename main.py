"""
Adapted from https://github.com/Amshaker/SwiftFormer
"""
import argparse
import datetime
import json
import numpy as np
import os
import time
import torch
import torch.utils.data
import uuid
import warnings

from collections import OrderedDict
from PIL import ImageFile

from models.swiftformer import SwiftFormer_XS

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from util.common import common_paths, is_office, init_office_args
from util.dataset import build_dataset
from util.engine import train_one_epoch, evaluate
from util.losses import DistillationLoss
from util.utils import init_session
from util.visualization import plot_learning_curves
from util.samplers import RASampler


ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", message='Palette images with Transparency expressed in bytes should be converted to RGBA images')


def get_args_parser():
    parser = argparse.ArgumentParser('SwiftFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    # Model parameters
    parser.add_argument('--model', default='SwiftFormer_XS', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224,type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.01, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='agc', help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.025, help='weight decay (default: 0.025)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate (default: 2e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT', help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL', help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default=common_paths['teacher_path'])
    parser.add_argument('--distillation-type', default='soft', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.7, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default=common_paths['dataset_root'], type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=20, type=int, help='number classes of your dataset')
    parser.add_argument('--output_dir', default=common_paths['train_runs'], help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):

    print(args)
    num_tasks, global_rank = init_session(args.output_dir)

    device = torch.device(args.device)
   
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if is_office(): # use subset for development
        dataset_train = torch.utils.data.Subset(dataset_train, torch.arange(0, 6 * args.batch_size))
        dataset_val = torch.utils.data.Subset(dataset_val, torch.arange(0, 2 * args.batch_size))

    sampler_train = RASampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True) 
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
        
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        distillation=(args.distillation_type != 'none'),
        pretrained=False,
        fuse=False,
    )

    model.to(device)
    
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters:', n_parameters)

    teacher_model = None
    if args.distillation_type != 'none':
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=1000, # TODO num_classes=20 see below 
            global_pool='avg',
        )

        checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])

        n_inputs = teacher_model.head.fc.in_features
        classifier = torch.nn.Sequential(OrderedDict([('fc1', torch.nn.Linear(n_inputs, 20))]))
        teacher_model.head.fc = classifier
        
        teacher_model.to(device)
        teacher_model.eval()

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)

    print(f"Start training for {args.epochs} epochs")  
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, 
            device, epoch, loss_scaler, args.clip_grad, args.clip_mode, 
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_path = f'{args.output_dir}/checkpoint.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device)         
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                    }

        if args.output_dir:
            with open(f'{args.output_dir}/log.txt', mode='a') as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    plot_learning_curves(args.output_dir, save_fig=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SwiftFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if is_office():
        init_office_args(args)

    if args.output_dir:
        session_id = '{date:%Y-%m-%d}__'.format(date=datetime.datetime.now()) + uuid.uuid4().hex
        args.output_dir = os.join(args.output_dir, session_id)

    main(args)