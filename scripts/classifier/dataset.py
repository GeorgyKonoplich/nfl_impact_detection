import random 
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import seed_everything


class ImpactDatasetRetriever(Dataset):
    def __init__(self, labels, image_dir, config, transforms=None, test=False):
        super().__init__()

        seed_everything(config.SEED)
        self.labels_impact = labels[labels.impact==2][labels.confidence>1][labels.visibility>0].reset_index(drop=True)
        self.labels_no_impact = labels[labels.impact==1].reset_index(drop=True)

        self.transforms = transforms
        self.image_dir = image_dir
        self.margin = config.margin
        self.label_smoothing = config.label_smoothing

        self.test = test
        if test:
            test_indices = np.random.permutation(len(self.labels_no_impact))[:len(self.labels_impact)]
            self.labels_no_impact = self.labels_no_impact.loc[test_indices].reset_index(drop=True)

    def __getitem__(self, index):
        #print(index, len(self.labels_impact))
        if index < len(self.labels_impact):
            sample = self.labels_impact.loc[index]
            label = 1
        else:
            if self.test:
                sample = self.labels_no_impact.loc[index - len(self.labels_impact)]
                label = 0
            else:
                l = len(self.labels_no_impact)
                sample = self.labels_no_impact.loc[random.randint(0, l-1)]
                label = 0

        image = cv2.imread(f'{self.image_dir}/{sample.image_name}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        x, y, w, h = sample[['x', 'y', 'w', 'h']].values

        crop_size = max(w, h) // 2
        crop_size += crop_size * self.margin
        #print(crop_size)
        crop_size = round(crop_size)

        center = [0,0]
        center[0] = x + w // 2 + crop_size
        center[1] = y + h // 2 + crop_size

        im_size = image.shape
        pad_image = np.zeros((im_size[0]+2*crop_size, im_size[1]+2*crop_size, im_size[2]))

        # copy img image into center of result image
        pad_image[crop_size:pad_image.shape[0]-crop_size, 
                crop_size:pad_image.shape[1]-crop_size] = image

        crop_img = pad_image[center[1]-crop_size:center[1]+crop_size,
                            center[0]-crop_size:center[0]+crop_size].astype('uint8')

        if self.transforms:
                sample = self.transforms(**{
                    'image': crop_img,
                    'label': label
                })         
        if self.label_smoothing > 0:
            sample['label'] = np.clip(sample['label'], self.label_smoothing, 1 - self.label_smoothing)         
        sample['label'] = torch.tensor((sample['label'],)) 
        return sample

    def __len__(self) -> int:
        return len(self.labels_impact) * 2


