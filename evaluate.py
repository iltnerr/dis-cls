import argparse
import torch

from models.swiftformer import SwiftFormer_XS
from timm.models import create_model
from util.common import common_paths
from util.dataset import build_dataset, INCIDENTS
from util.engine import evaluate, evaluate_cm


def get_args_parser():
    parser = argparse.ArgumentParser('SwiftFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)

    # Model parameters
    parser.add_argument('--model', default='SwiftFormer_XS', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224,type=int, help='images input size')

    # Dataset parameters
    parser.add_argument('--data-path', default=common_paths['dataset_root'], type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=20, type=int, help='number classes of your dataset')
    parser.add_argument('--output_dir', default=common_paths['train_runs'], help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):

    device = torch.device(args.device)    
    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        num_classes=args.nb_classes
    )

    ckpt_path = common_paths['ckpt_best']
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'],strict=False)
    model.to(device)

    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    # TODO: process the dataset only once. currently 'evaluate' and 'evaluate_cm' require independent forward passes.
    evaluate_cm(model, data_loader_val, ckpt_path, device,
                class_indict=INCIDENTS,
                output_dir='output') 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SwiftFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
