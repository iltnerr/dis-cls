import os

from PIL import Image
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


Image.MAX_IMAGE_PIXELS = None

INCIDENTS = {
    "0": "airplane_accident",
    "1": "burned",
    "2": "car_accident",
    "3": "collapsed",
    "4": "drought",
    "5": "dust_sandstorm",
    "6": "earthquake",
    "7": "flooded",
    "8": "hailstorm",
    "9": "heavy_rainfall",
    "10": "ice_storm",
    "11": "landslide",
    "12": "oil_spill",
    "13": "on_fire",
    "14": "ship_boat_accident",
    "15": "snow_covered",
    "16": "thunderstorm",
    "17": "tornado",
    "18": "volcanic_eruption",
    "19": "wildfire"
}


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 20
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # This should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # Replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
