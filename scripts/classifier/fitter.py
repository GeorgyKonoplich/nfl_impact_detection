import sys

import torch
import os
from datetime import datetime
import time
from glob import glob
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.nn.modules.loss import BCEWithLogitsLoss


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

    def __init__(self, model, device, config):
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

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        reduction = "mean"
        self.loss_fn = BCEWithLogitsLoss(reduction=reduction)
        

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
                for group in self.optimizer.param_groups:
                    group['lr'] = self.config.lr
            self.load(self.restore_path, resume_training=resume_training)
            self.log(f"\n[RESTORED FROM]: {self.restore_path}")
        self.log(f"\n[AUGMENTATIONS]: {str(train_loader.dataset.transforms[:-1])}")
        

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, impact_loss, no_impact_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, \
                impact_loss: {impact_loss.avg:.5f}, no_impact_loss: {no_impact_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, impact_loss, no_impact_loss, accuracy = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, accuracy: {accuracy.avg}, \
                 impact_loss: {impact_loss.avg:.5f}, no_impact_loss: {no_impact_loss.avg:.5f}, time: {(time.time() - t):.5f}')


            self.best_summary_loss = summary_loss.avg
            self.model.eval()
            self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

            if not self.cosine_annealing and self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_impact_loss = AverageMeter()
        summary_no_impact_loss = AverageMeter()
        summary_accuracy = AverageMeter()
        t = time.time()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if self.config.verbose:
                    if step % self.config.verbose_step == 0:
                        print(
                            f'Val Step {step}/{len(val_loader)}, ' + \
                            f'summary_loss: {summary_loss.avg:.5f}, ' + \
                            f'time: {(time.time() - t):.5f}', end='\r'
                        )

                imgs = batch["image"].to(self.device)
                labels = batch["label"].float().cpu()
                out_labels = self.model(imgs).cpu()

                impact_loss = 0
                no_impact_loss = 0
                impact_idx = labels > 0.5
                no_impact_idx = labels <= 0.5

                if torch.sum(impact_idx * 1) > 0:
                    impact_loss = self.loss_fn(out_labels[impact_idx], labels[impact_idx])
                if torch.sum(no_impact_idx * 1) > 0:
                    no_impact_loss = self.loss_fn(out_labels[no_impact_idx], labels[no_impact_idx])

                loss = (impact_loss + no_impact_loss) / 2

                #print('labels',labels)
                #print('preds', preds)

                #labels = labels.cpu().numpy()
                preds = torch.round(torch.sigmoid(out_labels)).numpy()
                
                #loss = self.loss_fn(out_labels, labels)
                
                labels = labels.flatten().numpy()
                labels[labels>=0.5] = 1 
                labels[labels<0.5] = 0
                preds = preds.flatten()
                #print('labels',labels)
                #print('preds', preds)
                accuracy = accuracy_score(labels.astype(int), preds.astype(int))
                
    
                summary_loss.update(loss.detach().item(), imgs.size(0))
                summary_impact_loss.update(impact_loss.detach().item(), imgs.size(0))
                summary_no_impact_loss.update(no_impact_loss.detach().item(), imgs.size(0))
                summary_accuracy.update(accuracy, imgs.size(0))

        return summary_loss, summary_impact_loss, summary_no_impact_loss, summary_accuracy

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        summary_impact_loss = AverageMeter()
        summary_no_impact_loss = AverageMeter()
        t = time.time()
        if self.cosine_annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(train_loader))
        pbar = tqdm(train_loader, desc='Epoch: ' + str(self.epoch))
        for step, batch in enumerate(pbar):
            try:
                pbar.set_description(f"summary_loss: {summary_loss.avg}")
                #,  \ impact_loss: {summary_impact_loss.avg},  \
                # no_impact_loss: {summary_no_impact_loss.avg}")
                
                

                self.optimizer.zero_grad()

                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

                out_labels = self.model(imgs)

                impact_loss = 0
                no_impact_loss = 0
                impact_idx = labels > 0.5
                no_impact_idx = labels <= 0.5

                if torch.sum(impact_idx * 1) > 0:
                    impact_loss = self.loss_fn(out_labels[impact_idx], labels[impact_idx])
                if torch.sum(no_impact_idx * 1) > 0:
                    no_impact_loss = self.loss_fn(out_labels[no_impact_idx], labels[no_impact_idx])

                loss = (impact_loss + no_impact_loss) / 2
    
                loss.backward()

                summary_loss.update(loss.detach().item(), imgs.size(0))
                summary_impact_loss.update(0 if impact_loss == 0 else impact_loss.detach().item(), imgs.size(0))
                summary_no_impact_loss.update(0 if no_impact_loss == 0 else no_impact_loss.detach().item(), imgs.size(0))

                self.optimizer.step()
                if self.cosine_annealing:
                    self.scheduler.step()
                if not self.cosine_annealing and self.config.step_scheduler:
                    self.scheduler.step()
            except Exception as error:
                self.log(f'Failed to train on step {step} with {error}')

        return summary_loss, summary_impact_loss, summary_no_impact_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path, resume_training=True):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        self.log(f"\n[TRAINING CLASSIFIER]: {config.dataset_name}")
        self.log(f"\n[DATASET]: {config.dataset_name}")
        self.log(f"[BATCH SIZE]: {config.batch_size}")
        self.log(f"[LR]: {config.lr}")
        self.log(f"[SAVE FOLDER]: {config.folder}")
        self.log(f"\n[MARGIN]: {config.margin}")
        
