"""

copyright 2020 jintian & yk

"""


import torch.nn as nn
from alfred.utils.log import logger as logging


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, C3_size, C4_size, feature_size=512):
        super(FeaturePyramidNetwork, self).__init__()

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_2 = nn.Conv2d(feature_size, feature_size *
                              2, kernel_size=2, stride=2)

    def forward(self, inputs):
        C3, C4 = inputs  # c3=38x512,c4=19x1024size

        P4_x = self.P4_1(C4)  # p4_1=downsample,p4_x=19
        P4_upsampled_x = self.P4_upsampled(P4_x)  # P4_upsampled_x = 38
        #logging.info('P4_upsampled_x: {}'.format(P4_upsampled_x.shape))

        P4_x = C3 + P4_upsampled_x  # p4_x:19 对应元素相加
        # P4_x = self.P4_2(P4_x) #P4_2(P4_x)为36size
        # logging.info('P4_upsampled_x: {}'.format(P4_upsampled_x.shape))

        P3_x = self.P3_1(C3)  # C3=1024通道经过P3_1降到512为p3_x跟C4通道相同
        # logging.info('P3_x: {}'.format(P3_x.shape))
        P3_x = P3_x + P4_x
        P3_x = self.P3_2(P3_x)
        return [P3_x, P4_x]
