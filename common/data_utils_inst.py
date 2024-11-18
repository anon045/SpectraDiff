import os
import yaml
import natsort
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
import random
from torchvision.utils import save_image, make_grid

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def build_transforms(config, is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(tuple(config['data']['resize']), antialias=True),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(tuple(config['data']['normalize_mean']), tuple(config['data']['normalize_std']))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(tuple(config['data']['resize']), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(tuple(config['data']['normalize_mean']), tuple(config['data']['normalize_std']))
        ])
    return transform

def build_subset(config, dataset, data_type):
    subset_len = config['loader'][data_type]['subset']
    if subset_len > len(dataset):
        subset_len = len(dataset)
        print(f"Warning: Subset length for {data_type} is larger than the dataset length. Using the full dataset.")
    return Subset(dataset, range(subset_len))

def process_inst_image(inst_image):
    inst_array = np.array(inst_image)
    if len(inst_array.shape) == 2:
        inst_array = np.expand_dims(inst_array, axis=0)
    else:
        inst_array = np.transpose(inst_array, (2, 0, 1))
    return torch.from_numpy(inst_array).long()

class InstPairDataset(Dataset):
    def __init__(self, data_root, data_type, transform, config_type, config):
        self.base_path = data_root
        self.data_type = data_type
        self.transform = transform
        self.config_type = config_type
        self.config = config
        self.rgb_dir = os.path.join(self.base_path, data_type, 'rgb')
        self.ir_dir = os.path.join(self.base_path, data_type, 'ir')
        self.inst_dir = os.path.join(self.base_path, data_type, 'inst')
        self.rgb_images = natsort.natsorted(os.listdir(self.rgb_dir))
        self.ir_images = natsort.natsorted(os.listdir(self.ir_dir))
        self.inst_images = natsort.natsorted(os.listdir(self.inst_dir))

        self.resize_nearest = transforms.Resize(tuple(config['data']['resize']), interpolation=transforms.InterpolationMode.NEAREST)

        if config_type == 'train':
            assert len(self.rgb_images) == len(self.ir_images) == len(self.inst_images), "All folders must contain the same number of images"
        
    def __len__(self):
        return len(self.rgb_images)

    def apply_common_transforms(self, rgb, ir, inst):
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            ir = TF.hflip(ir)
            inst = TF.hflip(inst)
            
        if random.random() > 0.5:
            rgb = TF.vflip(rgb)
            ir = TF.vflip(ir)
            inst = TF.vflip(inst)
        return rgb, ir, inst

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.rgb_images[idx])
        ir_path = os.path.join(self.ir_dir, self.ir_images[idx])
        inst_path = os.path.join(self.inst_dir, self.inst_images[idx])

        rgb_image = Image.open(rgb_path).convert("RGB")
        ir_image = Image.open(ir_path).convert("RGB")
        inst_image = Image.open(inst_path)

        if self.config_type == 'train':
            rgb_image, ir_image, inst_image = self.apply_common_transforms(rgb_image, ir_image, inst_image)

        if self.transform:
            rgb_image = self.transform(rgb_image)
            ir_image = self.transform(ir_image)
            inst_image = self.resize_nearest(inst_image)

        inst_tensor = process_inst_image(inst_image)

        return rgb_image, ir_image, inst_tensor

def build_loader_inst(config):
    loaders = {}
    data_types = config['loader']['target']
    config_type = config['config_type']
    data_root = config['data']['root']

    for data_type in data_types:
        transform = build_transforms(config, is_train=(data_type == 'train'))
        dataset = InstPairDataset(data_root, data_type, transform, config_type, config)
        print(f'{data_type} loader generated:', len(dataset))

        if config['loader'][data_type]['subset']:
            dataset = build_subset(config, dataset, data_type)
            print(f'subset applied for {data_type}:', len(dataset))

        loader = DataLoader(
            dataset, 
            batch_size=config['loader'][data_type]['batch_size'],
            shuffle=config['loader'][data_type]['shuffle'],
            num_workers=8,
            pin_memory=True
        )
        loaders[data_type] = loader

    return loaders
