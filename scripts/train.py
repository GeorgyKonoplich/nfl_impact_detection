import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import pandas as pd
import random
import torch

from fire import Fire
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler


from nfl_impact_detection.scripts.model import get_net
from nfl_impact_detection.scripts.dataset import DatasetRetriever
from nfl_impact_detection.scripts.fitter import Fitter
from nfl_impact_detection.scripts.transform import get_train_transforms, get_valid_transforms



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainGlobalConfig:
    #restore_from = '../experiments/effdet5-cutmix-mixup_dataset_only_impact/best-checkpoint-005epoch.bin'
    #resume_training = True
    num_workers = 4
    batch_size = 3
    n_epochs = 10
    lr = 5e-4
    folder = '../experiments/effdet5_ds_only_impact_master_22_12_2020_'
    dataset_name = 'dataset_only_impact'
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = False
    soft_nms = True
    cosine_annealing = True
    #SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    #SchedulerClass =  torch.optim.lr_scheduler.CosineAnnealingLR
    """
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    """


def collate_fn(batch):
    return tuple(zip(*batch))

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
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=SequentialSampler(validation_dataset),
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=config, evaluation_labels_path='../data/evaluation_labels.csv')
    fitter.fit(train_loader, val_loader)


def main(**kwargs):
    SEED = 42
    seed_everything(SEED)

    config = TrainGlobalConfig
    

    train_video_labels = pd.read_csv(f'../data/{config.dataset_name}/train_video_labels.csv')
    val_video_labels = pd.read_csv(f'../data/{config.dataset_name}/val_video_labels.csv')
    image_dir = '../data/train_images_all'

    train_dataset = DatasetRetriever(
        labels=train_video_labels,
        transforms=get_train_transforms(),
        image_dir=image_dir,
        test=False
    )

    validation_dataset = DatasetRetriever(
        labels=val_video_labels,
        transforms=get_valid_transforms(),
        image_dir=image_dir,
        test=True
    )

    net = get_net(config)

    run_training(net=net,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                config=TrainGlobalConfig)

if __name__ == '__main__':
    Fire(main)

