import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class DatasetRetriever(Dataset):

    def __init__(self, labels, image_dir, transforms=None, apply_cutmix_mixup=True):
        super().__init__()

        self.image_ids = labels.image_name.unique()
        #print(self.image_ids)
        self.labels = labels
        self.transforms = transforms
        self.image_dir = image_dir
        self.apply_cutmix_mixup = apply_cutmix_mixup

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if not self.apply_cutmix_mixup or random.random() > 0.5:
            output = self.load_image_and_boxes(index)
        elif random.random() > 0.5:
            output = self.load_cutmix_image_and_boxes(index)
        else:
            output = self.load_mixup_image_and_boxes(index)

        
        if output is None:
            return None
        image, boxes, labels = output
       
        target = {}
        target['boxes'] = boxes
        #target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    break
        target['labels'] = torch.tensor(sample['labels'])
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        try:
            image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0
            records = self.labels[self.labels['image_name'] == image_id]
            boxes = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            labels = records['impact'].values
        except:
            return None
        return image, boxes, labels

    def load_mixup_image_and_boxes(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        #(np.vstack((boxes, r_boxes)).astype(np.int32).shape)
        #print(np.concatenate((labels,r_labels), axis=None).shape)
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels,r_labels), axis=None)

    def load_cutmix_image_and_boxes(self, index, imsize=(1280, 720)):
        while True:
            w,h = imsize
            xc, yc = [int(random.uniform( h * 0.75, w * 0.25)) for _ in range(2)]  # center x, y
            indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

            result_image = np.full((h, w, 3), 1, dtype=np.float32)
            result_boxes = []
            result_labels = np.array([], dtype=np.int)

            for i, index in enumerate(indexes):
                image, boxes, labels = self.load_image_and_boxes(index)
                if i == 0:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w), min(h, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h) 

                result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b
                boxes[:, 0] += padw
                boxes[:, 1] += padh
                boxes[:, 2] += padw
                boxes[:, 3] += padh

                result_boxes.append(boxes)
                result_labels = np.concatenate((result_labels, labels))

            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0], 0, w, out=result_boxes[:, 0])
            np.clip(result_boxes[:, 2], 0, w, out=result_boxes[:, 2])
            np.clip(result_boxes[:, 3], 0, h, out=result_boxes[:, 3])
            np.clip(result_boxes[:, 1], 0, h, out=result_boxes[:, 1])

            result_boxes = result_boxes.astype(np.int32)
            index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
            result_boxes = result_boxes[index_to_use]
            result_labels = result_labels[index_to_use]
            
            if 2 in result_labels:
                all_boxes_far_from_edges = True
                for i, label in enumerate(result_labels):
                    box = result_boxes[i]
                    eps = 0.07
                    if label==2 and (abs(w-box[0])<eps*w or abs(w-box[2])<eps*w \
                                or abs(h-box[1])<eps*h or abs(h-box[3])<eps*h  \
                                or abs(box[0])<eps*w or abs(box[2])<eps*w \
                                or abs(box[1])<eps*h or abs(box[3])<eps*h):
                        #print(box)
                        all_boxes_far_from_edges = False
                        #print(box)
                
                #print(result_boxes)
                if all_boxes_far_from_edges:
                    break
        return result_image, result_boxes, result_labels
    """
    def load_cutmix_image_and_boxes(self, index, width=1280, height=720):
        #
        # This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        # Refactoring and adaptation: https://www.kaggle.com/shonenkov
        #
        w, h = width, height
        imsize = 1024
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            #print(image.shape)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_labels = result_labels[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes, result_labels
    """