import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import torch
import tqdm
from scipy.optimize import linear_sum_assignment


VAL_ROOT_PATH = '../data/train_images_all'


def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes):
    cost_matix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0]-box2[0])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[1:])

            if iou_score < 0.35:
                continue
            else:
                cost_matix[i,j]=0

    row_ind, col_ind = linear_sum_assignment(cost_matix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp=0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i,j]==0:
            tp+=1
        else:
            fp+=1
            fn+=1
    return tp, fp, fn

class Validator:

    def __init__(self, marking, device, transforms=None):
        self.epoch = 0
        self.device = device
        self.video_names = marking.video.unique()
        images_valid_dict = {}
        for vid_name in self.video_names:
            images_valid_dict[vid_name] = marking[marking.video == vid_name].image_name.unique().tolist()

        self.images_valid_dict = images_valid_dict
        self.transforms = transforms
        self.marking = marking

    def get(self, video_name: str, index: int):
        image_id = self.images_valid_dict[video_name][index]

        image, boxes, labels = self.load_image_and_boxes(image_id)

        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
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
        return image, target, image_id

    def load_image_and_boxes(self, image_id):
        image = cv2.imread(f'{VAL_ROOT_PATH}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_name'] == image_id]
        records.loc[(records.impact==2) & (records.confidence<=1) & (records.visibility==0), 'impact']=1
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['impact'].values
        return image, boxes, labels

    def predict(self, model, image, target, score_threshold=0.5):
        box_list = []
        score_list = []
        with torch.no_grad():
            batch_size = 1
            image = image.to(self.device).float()[None, :]
            target_res = {}
            target_res['bbox'] = [target['boxes'].to(self.device).float()]
            target_res['cls'] = [target['labels'].to(self.device).float()]
            target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
            target_res["img_size"] = torch.tensor([image[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)
            outputs = model(image, target_res)['detections']
            #print(outputs.shape)
            for i in range(batch_size):
                boxes = outputs[i].detach().cpu().numpy()[:, :4]
                scores = outputs[i].detach().cpu().numpy()[:, 4]
                label = outputs[i].detach().cpu().numpy()[:, 5]
                indexes = np.where((scores > score_threshold) & (label == 2))[0]
                box_list.append(boxes[indexes])
                score_list.append(scores[indexes])
        return box_list, score_list


    def calculate_metrics(self, model, score_threshold=0.4):
        ftp, ffp, ffn = [], [], []
        model.eval()
        with torch.no_grad():
            for vid_name in tqdm.tqdm(self.video_names):
                vid_frame_names = self.images_valid_dict[vid_name]
                num_frames = len(vid_frame_names)
                pred_boxes = []
                gt_boxes = []

                for ind in range(num_frames):
                    image, target, image_ids = self.get(vid_name, ind)
                    box_list, score_list = self.predict(model, image, target, score_threshold)

                    boxes = box_list[0].astype(np.int32).clip(min=0, max=511)
                    scores = score_list[0]

                    if len(scores) >= 1:
                        for j, box in enumerate(boxes):
                            box[0] = box[0] * 1280 / 512
                            box[1] = box[1] * 720 / 512
                            box[2] = box[2] * 1280 / 512
                            box[3] = box[3] * 720 / 512
                            boxes[j] = box
                    frame_idx = int(image_ids[image_ids.rfind('_') + 1:image_ids.rfind('.')])
                    pred_boxes += [[frame_idx] + list(box) for box in boxes if len(box) > 0]

                    boxes = list(target['boxes'].float().numpy()[target['labels'].float().numpy() == 2])
                    for j, box in enumerate(boxes):
                        box[[0, 1, 2, 3]] = box[[1, 0, 3, 2]]
                        box[0] = box[0] * 1280 / 512
                        box[1] = box[1] * 720 / 512
                        box[2] = box[2] * 1280 / 512
                        box[3] = box[3] * 720 / 512

                        boxes[j] = [frame_idx] + list(box)

                    gt_boxes += boxes
                tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
                ftp.append(tp)
                ffp.append(fp)
                ffn.append(fn)

        tp = np.sum(ftp)
        fp = np.sum(ffp)
        fn = np.sum(ffn)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        return tp, fp, fn, precision, recall, f1_score

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1