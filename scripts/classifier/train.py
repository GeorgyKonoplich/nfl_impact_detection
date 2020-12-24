from timm.models.efficientnet import tf_efficientnet_b4_ns, tf_efficientnet_b3_ns, \
    tf_efficientnet_b5_ns, tf_efficientnet_b2_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import cv2
import os
import random
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.nn.modules.loss import BCEWithLogitsLoss
from fire import Fire 

from fitter import Fitter
from model import ImpactClassifier
from dataset import ImpactDatasetRetriever
from utils import seed_everything


class TrainClassifierConfig:
    #network = "DeepFakeClassifier"
    #restore_from = '../experiments/effdet5_ds_only_impact_master_23_12_2020/best-checkpoint-011epoch.bin'
    #resume_training = True
    num_workers = 4
    batch_size = 32
    n_epochs = 30
    lr = 5e-4
    folder = '../../experiments/classifiers/effnet7_baseline'

    dataset_name = 'dataset_only_impact'
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = False
    cosine_annealing = True
    label_smoothing = 0.01
    margin = 2.5

    SEED = 42
    encoder = {
                "name" : tf_efficientnet_b7_ns,
                "features": 2560
              }
    
    
    optimizer = {
        "type": "SGD",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "learning_rate": 0.01,
        "nesterov": True,
        "schedule": {
            "type": "poly",
            "mode": "step",
            "epochs": 40,
            "params": {"max_iter":  100500}
        }
    }
    





#scheduler = create_optimizer(conf['optimizer'], model)
def get_train_transforms():
    return A.Compose(
        [
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.05),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.FancyPCA(),
                A.HueSaturationValue()
              ], p=0.7),
            
            #A.OneOf([
            #    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
            #                         val_shift_limit=0.2, p=0.9),
            #    A.RandomBrightnessContrast(brightness_limit=0.2, 
            #                         contrast_limit=0.2, p=0.9),
            #],p=0.9),
            A.ToGray(p=0.2),
            A.HorizontalFlip(p=0.5),
            #A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0)
            ],p=0.1),
            A.Resize(height=128, width=128, p=1),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )



def get_valid_transforms():
    return A.Compose(
        [   
            A.Resize(height=128, width=128, p=1),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )



def run_training(net, train_dataset, validation_dataset, config):
    device = torch.device('cuda:0')
    net.to(device)
    #net = torch.nn.DataParallel(net).to(device)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
        #collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size * 2,
        sampler=RandomSampler(validation_dataset),
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False,
        #collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=config)
    fitter.fit(train_loader, val_loader)


def main(**kwargs):
    

    config = TrainClassifierConfig
    seed_everything(config.SEED)

    net = ImpactClassifier(encoder=config.encoder)


    #optimizer = optim.Adam(model.parameters(),
    #                    lr=config.optimizer["learning_rate"],
    #                    weight_decay=config.optimizer["weight_decay"])
    

    
    train_video_labels = pd.read_csv(f'../../data/{config.dataset_name}/train_video_labels.csv', index_col=0)
    val_video_labels = pd.read_csv(f'../../data/{config.dataset_name}/val_video_labels.csv',  index_col=0)
    image_dir = '../../data/train_images_all'

    
    train_dataset = ImpactDatasetRetriever(train_video_labels, image_dir, config, transforms=get_train_transforms(), test=False)
    validation_dataset = ImpactDatasetRetriever(val_video_labels, image_dir, config, transforms=get_valid_transforms(), test=True)

    run_training(net=net,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                config=config)

if __name__ == '__main__':
    Fire(main)
