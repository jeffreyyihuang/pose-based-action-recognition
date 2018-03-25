from .pose_estimator import PoseEstimator
from resnet_3d import resnet18
import torch.nn as nn


class EndtoEnd3DPoseStream():
    def __init__(self, pose_weight):
        heatmap_extractor = PoseEstimator(pose_weight)
        res3d18 = resnet18(pretrained=True, num_classes=15)
        
        self.conv_channe_wise = nn.Conv3d(18, 1, kernel_size=(3, 1, 1), 
                                          stride=1, bias=False)