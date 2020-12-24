import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import pickle
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

VAL_ROOT_PATH = '../data/train_images_all'


def nms(dets, thresh=0.5, fr_th=200):
    x1 = dets[:, 2]
    y1 = dets[:, 3]
    x2 = dets[:, 4]
    y2 = dets[:, 5]
    scores = dets[:, 0]
    frames = dets[:, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        frm = frames[order[1:]]

        inds = np.where((ovr <= thresh) | (abs(frm - frames[i]) > fr_th))[0]
        # inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


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
            dist = abs(box1[0] - box2[1])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[2:])

            if iou_score < 0.35:
                continue
            else:
                cost_matix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def predict_boxes_for_validation_videos(validator, model, num_val_videos=None):
    model.eval()
    if num_val_videos is None:
        num_val_videos = len(validator.video_names)

    for vid_name in tqdm(validator.video_names[:num_val_videos]):
        vid_frame_names = validator.images_valid_dict[vid_name]
        num_frames = len(vid_frame_names)
        pred_boxes = []
        gt_boxes = []

        for ind in range(num_frames):
            image, target, image_ids = validator.get(vid_name, ind)
            box_list, score_list = validator.predict(model, image, target)

            boxes = box_list[0].astype(np.int32).clip(min=0, max=511)
            scores = score_list[0]

            frame_idx = int(image_ids[image_ids.rfind('_') + 1:image_ids.rfind('.')])
            pred_boxes += [[score, frame_idx] + list(box) for box, score in zip(boxes, scores) if len(box) > 0]

            boxes = list(target['boxes'].float().numpy()[target['labels'].float().numpy() == 2])
            for j, box in enumerate(boxes):
                box[[0, 1, 2, 3]] = box[[1, 0, 3, 2]]
                boxes[j] = [frame_idx] + list(box)

            gt_boxes += boxes
        yield vid_name, pred_boxes, gt_boxes


def get_metrics(video2boxes, score_threshold=0.4, nms_threshold=None):
    ftp, ffp, ffn = [], [], []
    for vid_name in video2boxes.keys():
        pred_filtered_boxes = [box for box in video2boxes[vid_name][0] if box[0] > score_threshold]
        if nms_threshold is not None and len(pred_filtered_boxes) > 0:
            indices = nms(np.array(pred_filtered_boxes), nms_threshold)
            pred_filtered_boxes = list(np.array(pred_filtered_boxes)[indices])
        tp, fp, fn = precision_calc(video2boxes[vid_name][1], pred_filtered_boxes)
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


def calculate_metrics(validator, model, score_threshold=0.4, nms_threshold=0.5, num_val_videos=5):
    video2boxes = {}
    for vid_name, pred_boxes, gt_boxes in predict_boxes_for_validation_videos(validator, model, num_val_videos):
        video2boxes[vid_name] = (pred_boxes, gt_boxes)

    tp, fp, fn, precision, recall, f1_score = get_metrics(video2boxes=video2boxes, score_threshold=score_threshold)
    log1 = f'Before nms, score_threshold: {score_threshold:.2f}, TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.2f}, RECALL: {recall:.2f}, F1 SCORE: {f1_score:.2f}'

    tp, fp, fn, precision, recall, f1_score = get_metrics(video2boxes=video2boxes,
                                                          score_threshold=score_threshold,
                                                          nms_threshold=nms_threshold)
    log2 = f'After nms, score_threshold: {score_threshold:.2f}, nms_threshold: {nms_threshold}, TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.2f}, RECALL: {recall:.2f}, F1 SCORE: {f1_score:.2f}'

    return log1, log2


class Validator:

    def __init__(self, labels, device, transforms=None):
        self.epoch = 0
        self.device = device
        self.video_names = labels.video.unique()
        images_valid_dict = {}
        for vid_name in self.video_names:
            images_valid_dict[vid_name] = labels[labels.video == vid_name].image_name.unique().tolist()

        self.images_valid_dict = images_valid_dict
        self.transforms = transforms
        self.labels = labels

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        records = self.labels[self.labels['image_name'] == image_id]
        records.loc[(records.impact == 2) & (records.confidence <= 1), 'impact'] = 1
        records.loc[(records.impact == 2) & (records.visibility == 0), 'impact'] = 1
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes, records['impact'].values

    def predict(self, model, image, target):
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
            for i in range(batch_size):
                boxes = outputs[i].detach().cpu().numpy()[:, :4]
                scores = outputs[i].detach().cpu().numpy()[:, 4]
                label = outputs[i].detach().cpu().numpy()[:, 5]
                indexes = np.where(label == 2)[0]
                box_list.append(boxes[indexes])
                score_list.append(scores[indexes])
        return box_list, score_list
