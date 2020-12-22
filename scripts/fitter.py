import sys

sys.path.insert(0, "timm-efficientdet-pytorch")
sys.path.insert(0, "omegaconf")

import torch
import os
from datetime import datetime
import time
from glob import glob
import pandas as pd
import shutil
from tqdm import tqdm

from nfl_impact_detection.scripts.transform import get_valid_transforms
from nfl_impact_detection.scripts.validator import Validator, calculate_metrics


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:

    def __init__(self, model, device, config, evaluation_labels_path):
        self.config = config
        self.epoch = 0
        self.restore_path = None
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if hasattr(config, 'restore_from'):
            self.restore_path = config.restore_from
            ind = self.restore_path.rfind('/')
            restore_log_path = self.restore_path[:ind]
            shutil.copy2(os.path.join(restore_log_path, 'log.txt'), self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        self.validator = Validator(device=device,
                                   labels=pd.read_csv(evaluation_labels_path),
                                   transforms=get_valid_transforms())

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        

        if config.cosine_annealing:
            self.cosine_annealing = True
        else:
            self.cosine_annealing = False
            self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        
            
        self.log_config(config)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        if self.restore_path:
            if hasattr(self.config, 'resume_training'):
                resume_training = self.config.resume_training
            self.load(self.restore_path, resume_training=resume_training)
            self.log(f"\n[RESTORED FROM]: {self.restore_path}")
        self.log(f"\n[AUGMENTATIONS]: {str(train_loader.dataset.transforms[:-1])}")

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            log1, log2 = calculate_metrics(validator=self.validator,
                                           model=self.model,
                                           score_threshold=0.4,
                                           nms_threshold=0.5,
                                           num_val_videos=5)
            self.log(log1)
            self.log(log2)

            self.best_summary_loss = summary_loss.avg
            self.model.eval()
            self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

            if not self.cosine_annealing and self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                target_res = {}
                target_res['bbox'] = boxes
                target_res['cls'] = labels
                target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
                target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(
                    self.device)

                outputs = self.model(images, target_res)
                loss = outputs['loss']
    
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        if self.cosine_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader))
        pbar = tqdm(train_loader, desc='Epoch: ' + str(self.epoch))
        for step, (images, targets, image_ids) in enumerate(pbar):
            try:
                pbar.set_description("summary_loss: %s" % summary_loss.avg)
                print(
                    f'Train Step {step}/{len(train_loader)},  summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
                # print(step)
                # if self.config.verbose:
                #    if step % self.config.verbose_step == 0:

                self.scheduler.step()

                images = torch.stack(images)
                images = images.to(self.device).float()
                batch_size = images.shape[0]
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                target_res = {}
                target_res['bbox'] = boxes
                target_res['cls'] = labels
                target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
                target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(
                    self.device)

                self.optimizer.zero_grad()

                # targets

                outputs = self.model(images, target_res)
                loss = outputs['loss']
    
                loss.backward()

                summary_loss.update(loss.detach().item(), batch_size)

                self.optimizer.step()

                if not self.cosine_annealing and self.config.step_scheduler:
                    self.scheduler.step()
            except Exception as error:
                self.log(f'Failed to train on step {step} with {error}')

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path, resume_training=True):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if not self.cosine_annealing:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_summary_loss = checkpoint['best_summary_loss']
            self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

    def log_config(self, config):
        self.log(f"\n[DATASET]: {config.dataset_name}")
        self.log(f"[BATCH SIZE]: {config.batch_size}")
        self.log(f"[LR]: {config.lr}")
        self.log(f"[SAVE FOLDER]: {config.folder}")
