import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def get_net(train_config):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.image_size = [512, 512]
    config.norm_kwargs = dict(eps=0.001, momentum=0.01)
    config.soft_nms = train_config.soft_nms
    net = EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load('../data/efficientdet_d5-ef44aea8.pth')
    net.load_state_dict(checkpoint)
    net.reset_head(num_classes=2)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config)