import json
from collections import Counter
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def count_colours(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    colours_list = [entry['colour'] for entry in data]
    return Counter(colours_list)

def weighted_bce_loss(output, target, white_thresh_low=0.96, white_thresh_high=1, device='cuda'):
    white_mask = (target >= white_thresh_low) & (target <= white_thresh_high)
    white_mask = white_mask.float().to(device)

    normal_mask = 1.0 - white_mask
    bce_per_pixel = F.binary_cross_entropy(output, target, reduction='none')
    loss = bce_per_pixel * white_mask + 1.2 * bce_per_pixel * normal_mask

    return loss.mean()

def extract_named_colors(json_path, base_path='/kaggle/input/ayna-unet-dataset/dataset/training/outputs'):
    with open(json_path, 'r') as f:
        data = json.load(f)

    color_map = {}

    for item in data:
        color_name = item['colour']
        img_path = item['output_image']

        img = Image.open(f'{base_path}/{img_path}').convert('RGB')
        img_np = np.array(img)

        unique_colors = np.unique(img_np.reshape(-1, 3), axis=0)
        normalized_colors = (unique_colors / 255).round(3)
        norm_color_tensor = torch.tensor(normalized_colors[0].tolist())

        if color_name not in color_map:
            color_map[color_name] = norm_color_tensor
        else:
            if not torch.allclose(color_map[color_name], norm_color_tensor):
                print(f"Warning: Multiple different colors found for '{color_name}'")

    return color_map
