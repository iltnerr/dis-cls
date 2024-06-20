"""
Attention Visualization
"""
import cv2
import numpy as np
import os 
import random
import torch

from models.swiftformer import SwiftFormer
from PIL import Image
from torchvision import transforms
from util.common import common_paths
from util.dataset import INCIDENTS
from util.visualization import get_attn_weights, visualize_attention


print(INCIDENTS)
ds_path = os.path.join(common_paths['dataset_root'], 'val', INCIDENTS['18'])
output_dir = 'output/' 
ckpt_path = common_paths['ckpt_best']

image_list = os.listdir(ds_path)
random.shuffle(image_list)
image_path = os.path.join(ds_path, random.choice(image_list))

model = SwiftFormer(
            layers=[3, 3, 3, 3],
            embed_dims=[32, 36,40, 48],
            downsamples=[True, True, True, True],
            vit_num=1)
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval() 

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

for image in image_list:
    image_path = os.path.join(ds_path, image)
    print(f"\n\n\n{image_path}\n")

    img = Image.open(image_path)
    img = img.resize((224, 224))
    t = transform(img)
    input_tensor = torch.unsqueeze(t,0)

    s = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        res = model(input_tensor)
        prob = s(res)
        print(torch.topk(prob, 3))
        
    # Global attention query vector for different stages
    attn_stages = [
        model.network[0][2].attn,    # stage 1
        model.network[2][2].attn,    # stage 2
        model.network[4][2].attn,    # stage 3
        model.network[6][2].attn     # stage 4
    ]

    weights = [get_attn_weights(stage) for stage in attn_stages]

    # fuse weights for all stages
    size = weights[0].shape
    agg = np.array([cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC) for arr in weights]) # all arrays same size

    fused_weights = [
        np.mean(agg, axis=0),
        np.max(agg, axis=0),
        np.min(agg, axis=0)
    ]

    # plot attention weights
    maps = {
        "Input Image": np.array(img), 
        "Mean": fused_weights[0], 
        "Stage 1": weights[0],
        "Stage 2": weights[1],
        "Stage 3": weights[2], 
        "Stage 4": weights[3] 
    }

    visualize_attention(maps)