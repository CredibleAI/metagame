import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
from datasets import load_dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image

import fixlip

# Path to a pre-extracted ImageNet-1k arrow dump (validation split). If unset,
# `load_dataset("imagenet-1k", split="validation")` pulls from HuggingFace
# (requires HF auth + EULA acceptance).
IMAGENET_PATH = os.environ.get("IMAGENET_PATH")
if IMAGENET_PATH:
    dataset = load_dataset(
        "arrow",
        data_files={"validation": f"{IMAGENET_PATH}/*validation*.arrow"},
        split="validation",
    )
else:
    dataset = load_dataset("imagenet-1k", split="validation")
print(f"dataset: {dataset}")

labels = {
    'fish': 0,       # Domain: Water Animal (From your list)
    'koala': 105,    # Domain: Land Animal (Furry, distinct from fish)
    'plane': 404,    # Domain: Air Vehicle (From your list, distinct sky context)
    'balloon': 417,  # Domain: Airborne Object (Smooth, non-rigid, distinct from plane)
    'church': 497,   # Domain: Architecture (From your list, strong structural features)
    'jeep': 609,     # Domain: Ground Vehicle (Boxy, distinct from planes)
    'laptop': 620,   # Domain: Electronics (Screens, keyboards, right angles)
    'lemon': 951,    # Domain: Raw Fruit (Textured, distinct yellow oval)
    'pizza': 963,    # Domain: Cooked Food (From your list, flat and distinct from lemon)
    'acorn': 988     # Domain: Nature/Seed (Woody, small, specific cap texture)
}

DATASET_ROOT = os.environ.get("POINTING_GAME_DATASET", "pointing_game")
for label, id in labels.items():
    idx_label = np.where(np.array(dataset['label']) == id)[0]
    dataset_label = dataset.select(idx_label)
    print(f"  {label} (id={id})  n={len(idx_label)}")
    path = f'{DATASET_ROOT}/{label}'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, item in enumerate(dataset_label):
        item['image'].save(f'{path}/{i}.jpg')

games = [
    ['fish', 'koala', 'balloon', 'laptop'],
    ['koala', 'plane', 'church', 'lemon'],
    ['plane', 'balloon', 'jeep', 'pizza'],
    ['balloon', 'church', 'laptop', 'acorn'],
    ['church', 'jeep', 'lemon', 'fish'],
    ['jeep', 'laptop', 'pizza', 'koala'],
    ['laptop', 'lemon', 'acorn', 'plane'],
    ['lemon', 'pizza', 'fish', 'balloon'],
    ['pizza', 'acorn', 'koala', 'church'],
    ['acorn', 'fish', 'plane', 'jeep']
]

resizer = T.Compose([
    T.Resize(224),       # Scales the shorter edge to 224, preserving aspect ratio
    T.CenterCrop(224)    # Crops the exact middle 224x224 square out of the longer edge
])

for game in games:
    cl = "_".join(game)
    path_cl = f'{DATASET_ROOT}/{cl}/'
    if not os.path.exists(path_cl):
        os.makedirs(path_cl)
    for i in range(50):
        images = []
        for label in game:
            img = Image.open(f'{DATASET_ROOT}/{label}/{i}.jpg')
            images.append(resizer(img))
        img1 = fixlip.utils.append_images([images[0], images[1]], direction='horizontal')
        img2 = fixlip.utils.append_images([images[2], images[3]], direction='horizontal')
        final = fixlip.utils.append_images([img1, img2], direction='vertical')
        final.save(f'{path_cl}/{i}.jpg')
